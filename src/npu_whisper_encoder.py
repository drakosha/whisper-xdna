#!/usr/bin/env python3
"""Whisper encoder with every GEMM dispatched to the AMD XDNA1 NPU.

Size-agnostic: layer count, model width and head count come from the weights.
Runs in ironenv (MLIR-AIE / IRON, numpy only -- no torch). Reads an .npz from
tools/dump_ref.py (weights + mel + torch reference output).

Backends:
  --backend numpy   pure numpy float32 (validates the implementation itself)
  --backend npu     every matmul -> AIE array, bf16 in / f32 accumulate+out
  --attn cpu|npu    whether the per-head attention GEMMs also go to the NPU

Everything that is not a GEMM (layernorm, GELU, softmax, residuals, im2col)
stays on the CPU in numpy float32.
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np

from repo_paths import ROOT

REF = str(ROOT / "ref.npz")
MLIR_AIE = os.environ.get("MLIR_AIE_DIR", "/opt/mlir-aie")
WA_PY = os.path.join(MLIR_AIE, "programming_examples/basic/"
                     "matrix_multiplication/whole_array/whole_array.py")
# The GEMM design with GELU fused into the core is not in this repo; --fused-gelu
# needs it pointed at with NPU_FUSED_PY. Every other path here works without it.
FUSED_PY = os.environ.get("NPU_FUSED_PY", "")

# ---------------------------------------------------------------- NPU backend

_npu = {}


def npu_init(n_aie_cols=4):
    import aie.iron as iron
    from aie.iron.device import from_name
    from aie.utils.hostruntime import set_current_device

    spec = importlib.util.spec_from_file_location("whole_array_mod", WA_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["whole_array_mod"] = mod
    spec.loader.exec_module(mod)

    set_current_device(from_name("npu", n_cols=n_aie_cols))
    import ml_dtypes
    _npu.update(iron=iron, design=mod.whole_array, fused=None,
                bf16=ml_dtypes.bfloat16,
                n_aie_cols=n_aie_cols, calls=0, npu_ns=0, e2e=0.0, shapes={})


def _fused_design():
    """The GEMM+GELU design, loaded on demand so its absence costs nothing."""
    if _npu.get("fused") is None:
        if not FUSED_PY or not os.path.exists(FUSED_PY):
            raise RuntimeError(
                "--fused-gelu needs the fused matmul design, which is not in "
                "this repo. Point NPU_FUSED_PY at a fused_mm_gelu.py that "
                "exports fused_mm_gelu(), or use --npu-gelu (rowops), which is "
                "the accurate path anyway.")
        fspec = importlib.util.spec_from_file_location("fused_mod", FUSED_PY)
        fmod = importlib.util.module_from_spec(fspec)
        sys.modules["fused_mod"] = fmod
        fspec.loader.exec_module(fmod)
        _npu["fused"] = fmod.fused_mm_gelu
    return _npu["fused"]


def _tiles(M, K, N):
    """Pick AIE tile sizes (m, k, n) for a GEMM shape.

    Measured on 4 columns, bf16: the default 64/64/32 collapses to ~110-160
    GFLOPS on the skinny attention shapes. Matching the tile to the small
    dimension recovers ~4x:
      (1536, 64, 1536)  64/64/32 -> 163 GFLOPS   32/64/64 -> 440 GFLOPS
      (1536, 1536, 64)  padded N=128 -> 159 eff. 64/64/16 -> 642 GFLOPS
    """
    if K <= 64:                 # Q @ K^T
        return 32, 64, 64
    if N <= 64:                 # attn_weights @ V
        return 64, 64, 16
    return 64, 64, 32


def _pad_to(v, mult):
    return ((v + mult - 1) // mult) * mult


def npu_matmul_gelu(A, B, bias, akey=None, bkey=None, out_into=None):
    """gelu(A @ B + bias), with the GELU fused into the matmul core.

    aie_kernels/aie2/gelu.cc runs on the (m, n) accumulator tile before it ever
    leaves the core, so the activation costs no extra DMA. The kernel sees only
    the matmul result, so the bias is folded into the GEMM: a constant-1 column
    on A and the bias as row K of B. That keeps gelu(A@B + b) rather than
    gelu(A@B) + b, and both are written into the padded operand buffers
    directly, so no augmented copy of A or B is ever built.

    out_into=(buf, r0, c0) writes the bf16 result straight into a slice of an
    existing bf16 buffer, so the fc1 chunks assemble fc2's input in place.
    """
    return _npu_gemm(A, B, fused_gelu=True, akey=akey, bkey=bkey,
                     bias=bias, out_into=out_into)


def npu_matmul(A, B, akey=None, bkey=None, out_into=None):
    """A (M,K) f32 @ B (K,N) f32 -> (M,N) f32, executed on the AIE array in bf16.

    akey/bkey name an operand that does not change between calls (weights, or an
    activation feeding several GEMMs like the layernorm output feeding q/k/v).
    Named operands are converted to bf16, padded and uploaded to a device buffer
    exactly once and then stay resident -- without this the host re-converts and
    re-uploads every weight on every call, which for one large-v3 block is 39 MB
    of f32->bf16 conversion 49 times over.
    """
    return _npu_gemm(A, B, fused_gelu=False, akey=akey, bkey=bkey,
                     out_into=out_into)


# Set to a list to collect fc1 pre-activations: the GELU polynomials in
# tools/gen_gelu*_coeffs.py are fitted on the REAL distribution (mean -1.99, a
# third of the mass in [-6, -3)), and without a tap there is no way to
# reproduce them from a clean checkout. Host paths only -- the device MLP never
# brings that tensor back. bench/run_raw_encoder.py --dump-fc1-acts drives it.
FC1_TAP = None


def _tap_fc1(z):
    if FC1_TAP is not None:
        FC1_TAP.append(np.asarray(z, dtype=np.float32).ravel()[::97].copy())
    return z


# Weights are invariant for the whole run; activations are only valid inside the
# block invocation that produced them. Keeping them in one dict would let a
# stale activation survive into the next block -- silently wrong results, and a
# benchmark that skips an upload it should be paying for.
_WCACHE = {}   # bkey -> device tensor, lives for the whole run
_ACACHE = {}   # akey -> device tensor, cleared at every block boundary


def clear_tensor_cache():
    _WCACHE.clear()
    _ACACHE.clear()


def new_block():
    """Invalidate cached activations. Call at the start of every block."""
    _ACACHE.clear()


def _operand(X, rows, cols, key, tag, ones_col=None, bias_row=None,
             cache=None):
    """Padded bf16 device tensor for X, cached under `key` when given.

    ones_col / bias_row fold a bias into the GEMM without materialising an
    augmented copy of either operand: the constant-1 column and the bias row are
    written straight into the padded buffer that has to be allocated anyway.

    Fast path: an operand that is already the exact padded bf16 buffer is handed
    to the device untouched, so a bf16 result produced by a previous GEMM feeds
    the next one with no conversion.
    """
    iron, bf16 = _npu["iron"], _npu["bf16"]
    if not isinstance(X, np.ndarray):
        # already a device tensor of exactly the padded shape: hand it straight
        # to the next kernel, no download and no re-upload
        return X
    ck = None if key is None else (key, tag, rows, cols)
    if ck is not None:
        t = cache.get(ck)
        if t is not None:
            return t
    t0 = time.perf_counter()
    if (X.dtype == bf16 and X.shape == (rows, cols)
            and X.flags["C_CONTIGUOUS"] and ones_col is None
            and bias_row is None):
        Xp = X
    else:
        Xp = np.zeros((rows, cols), dtype=bf16)
        Xp[:X.shape[0], :X.shape[1]] = X if X.dtype == bf16 else X.astype(bf16)
        if ones_col is not None:
            Xp[:X.shape[0], ones_col] = 1.0
        if bias_row is not None:
            idx, vec = bias_row
            Xp[idx, :vec.shape[0]] = vec if vec.dtype == bf16 else vec.astype(bf16)
    t = iron.tensor(Xp, dtype=bf16, device="npu")
    CPU_PROF["h2d_convert_upload"] = CPU_PROF.get("h2d_convert_upload", 0.0) + (
        time.perf_counter() - t0)
    if ck is not None:
        cache[ck] = t
    return t


def _npu_gemm(A, B, fused_gelu=False, akey=None, bkey=None, bias=None,
              out_into=None):
    iron, design, bf16 = _npu["iron"], _npu["design"], _npu["bf16"]
    if fused_gelu:
        design = _fused_design()
    nc = _npu["n_aie_cols"]
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    Kfold = K + 1 if bias is not None else K
    if fused_gelu:
        # gelu.cc hard-codes 1024 elements, so the C tile must be 32x32
        TM, TK, TN = 32, 64, 32
    else:
        TM, TK, TN = _tiles(M, K, N)
    # M/m/n_aie_rows must be a multiple of 2  ->  M % (TM*4*2) == 0
    Mp = _pad_to(M, TM * 4 * 2)
    Kp = _pad_to(Kfold, TK)
    Np = _pad_to(N, TN * nc)

    A_t = _operand(A, Mp, Kp, akey, "A",
                   ones_col=(K if bias is not None else None), cache=_ACACHE)
    B_t = _operand(B, Kp, Np, bkey, "B",
                   bias_row=((K, bias) if bias is not None else None),
                   cache=_WCACHE)
    out_dt = bf16 if fused_gelu else np.float32
    C_t = iron.zeros((Mp, Np), dtype=out_dt, device="npu")

    t0 = time.perf_counter()
    kw = dict(M=Mp, K=Kp, N=Np, m=TM, k=TK, n=TN, n_aie_cols=nc,
              dtype_in_str="bf16",
              dtype_out_str="bf16" if fused_gelu else "f32")
    if fused_gelu:
        kw["epilogue"] = "gelu"
    ret = design(A_t, B_t, C_t, **kw)
    e2e = time.perf_counter() - t0

    _npu["calls"] += 1
    _npu["e2e"] += e2e
    npu_ns = getattr(ret[1] if isinstance(ret, tuple) else ret, "npu_time", None)
    if npu_ns:
        _npu["npu_ns"] += npu_ns
    key = (Mp, Kp, Np, f"{TM}/{TK}/{TN}" + ("+gelu" if fused_gelu else ""))
    # design-id trace: a "switch" is a call whose design differs from the
    # previous NPU call. Counting these is the only way to know how much of the
    # per-block gap reconfiguration can actually account for.
    seq = _npu.setdefault("design_seq", [])
    if not seq or seq[-1] != key:
        _npu["design_switches"] = _npu.get("design_switches", 0) + 1
    seq.append(key)
    s = _npu["shapes"].setdefault(key, [0, 0.0, 0])
    s[0] += 1
    s[1] += e2e
    s[2] += npu_ns or 0
    # NOTE: C_t.numpy() is a view onto the XRT buffer object. C_t is a local,
    # so the BO is freed on return and the view dangles -> segfault at the next
    # numpy op that touches it. Copy out before returning.
    t1 = time.perf_counter()
    view = C_t.numpy().reshape(Mp, Np)[:M, :N]
    if out_into is not None:
        buf, r0, c0 = out_into
        # writing straight into the destination slice avoids materialising a
        # per-chunk array and the concatenate that used to follow
        buf[r0:r0 + M, c0:c0 + N] = view
        r = None
    else:
        r = view.astype(np.float32)
    CPU_PROF["dev2host_copy"] = CPU_PROF.get("dev2host_copy", 0.0) + (
        time.perf_counter() - t1)
    return r


def np_matmul(A, B, bias=None, **kw):
    r = A.astype(np.float32) @ B.astype(np.float32)
    return r if bias is None else r + bias


# ---------------------------------------------------------------- encoder ops

def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    v = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(v + eps) * w + b


def gelu(x):
    # torch nn.GELU() default is the exact erf formulation
    from math import sqrt
    return 0.5 * x * (1.0 + _erf(x / sqrt(2.0)))


def _erf(x):
    # numpy has no erf; Abramowitz-Stegun 7.1.26 is only ~1e-7 accurate, which
    # is fine relative to bf16 but would pollute the numpy-vs-torch check.
    # scipy is not in ironenv, so use the tanh-free high accuracy rational form.
    a1, a2, a3, a4, a5, p = (0.254829592, -0.284496736, 1.421413741,
                             -1.453152027, 1.061405429, 0.3275911)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + p * ax)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-ax * ax)
    return sign * y


CPU_PROF = {}
NEG = -30000.0  # bf16-safe stand-in for -inf on padded softmax columns
_ZERO_BIAS = {}


def zero_bias(d):
    """A stable all-zero bias row. Whisper's key projection has no bias, but the
    bias-folded GEMM design still needs a row for it -- and the array has to be
    the SAME object every call, or the weight cache would re-upload B."""
    b = _ZERO_BIAS.get(d)
    if b is None:
        b = _ZERO_BIAS[d] = np.zeros(d, dtype=np.float32)
    return b


def npu_gelu_dev(buf):
    """GELU over a whole bf16 buffer in one launch; returns the DEVICE tensor so
    the next GEMM consumes it without a download/upload round trip."""
    import rowops
    bf16 = _npu["bf16"]
    rows, cols = buf.shape
    a_t = _npu["iron"].tensor(buf, dtype=bf16, device="npu")
    c_t = _npu["iron"].zeros((rows, cols), dtype=bf16, device="npu")
    t0 = time.perf_counter()
    rowops.gelu_row(a_t, c_t, rows=rows, cols=cols)
    CPU_PROF["npu_gelu"] = CPU_PROF.get("npu_gelu", 0.0) + (
        time.perf_counter() - t0)
    return c_t


def npu_softmax_stack(mats):
    """Correct row-wise softmax for every head of a layer in one NPU launch.

    Rows are padded out to a multiple of 16 (the kernel's vector width) with a
    large negative value, so the padding contributes exp(x-max)~0 and does not
    change the normalisation.
    """
    import aie.iron as iron
    import rowops
    bf16 = _npu["bf16"]
    nh = len(mats)
    L0, L1 = mats[0].shape
    Lp = ((L1 + 15) // 16) * 16
    rows = nh * L0
    assert rows % 8 == 0, f"rows {rows} must be divisible by 8 cores"
    big = np.full((rows, Lp), NEG, dtype=bf16)
    for i, m in enumerate(mats):
        big[i * L0:(i + 1) * L0, :L1] = m.astype(bf16)
    a_t = iron.tensor(big, dtype=bf16, device="npu")
    c_t = iron.zeros((rows, Lp), dtype=bf16, device="npu")
    t0 = time.perf_counter()
    rowops.softmax_row(a_t, c_t, rows=rows, cols=Lp)
    CPU_PROF["npu_softmax"] = CPU_PROF.get("npu_softmax", 0.0) + (
        time.perf_counter() - t0)
    out = c_t.numpy().reshape(rows, Lp).astype(np.float32)
    return [out[i * L0:(i + 1) * L0, :L1] for i in range(nh)]


MAX_GEMM_N = 4096   # widest f32 GEMM output that compiles, see below


def prepare_mlp_chunks(W, n_layer, d, chunk=0):
    """Split every too-wide fc1 weight into column chunks, in place. Returns the
    new arrays so the caller can register them as resident weights.

    Needed because an f32 GEMM with N=5120 does not compile:

        error: 'aie.dma_bd' op Stride 3 exceeds the [1:1048576] range.
          aie.dma_bd(%arg2 : memref<7864320xf32>, 0, 327680,
            [<size = 2, stride = 1310720>, ...])

    The output descriptor's outer stride is (rows per block) * N = 256 * N, so
    N caps at 4096. Splitting by output column is exact: column j of A@B depends
    only on column j of B.

    Chunk width defaults to the model width d whenever d divides N (large-v3:
    5120 -> 4x1280), because that is the width q/k/v already compiled, so the
    MLP costs no extra overlay and no extra hw_context against a ceiling of 6.
    Anything that already compiles is left alone: tiny's fc1 is 1536 wide and
    must NOT be chunked, or it grows a second overlay for nothing.
    """
    made = []
    for L in range(n_layer):
        p = f"w.blocks.{L}."
        W1, b1 = W[p + "mlp.0.weight_T"], W[p + "mlp.0.bias"]
        N = W1.shape[1]
        if chunk:
            width = chunk
        elif N <= MAX_GEMM_N:
            continue
        elif N % d == 0:
            width = d
        else:
            width = N // ((N + MAX_GEMM_N - 1) // MAX_GEMM_N)
        if N <= width:
            continue
        parts = []
        for j in range(0, N, width):
            wc = np.ascontiguousarray(W1[:, j:j + width])
            bc = np.ascontiguousarray(b1[j:j + width])
            parts.append((wc, bc))
            made += [wc, bc]
        W[p + "mlp.0.parts"] = parts
    return made


def _timed(name, fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    CPU_PROF[name] = CPU_PROF.get(name, 0.0) + (time.perf_counter() - t0)
    return r


def softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def im2col(x, kernel, stride, pad):
    """x (C, T) -> (T_out, C*kernel), matching conv1d weight.reshape(O, C*k)."""
    C, T = x.shape
    xp = np.zeros((C, T + 2 * pad), dtype=x.dtype)
    xp[:, pad:pad + T] = x
    T_out = (T + 2 * pad - kernel) // stride + 1
    cols = np.empty((T_out, C * kernel), dtype=x.dtype)
    for j in range(kernel):
        cols[:, j::kernel] = xp[:, j:j + (T_out - 1) * stride + 1:stride].T
    return cols


# ---------------------------------------------------------------- the encoder

def encoder(W, mel, mm, attn_mm, n_layer=4, n_head=6, prof=None,
            fused_gelu=False, npu_softmax=False, attn_fn=None,
            mlp_fn=None, fold_bias=True):
    def tic():
        return time.perf_counter()

    t_all = tic()
    # --- conv frontend, both convs as im2col GEMMs
    c1w = W["w.conv1.weight"]                    # (384, 80, 3)
    O, C, KS = c1w.shape
    A = _timed("im2col", im2col, mel, KS, 1, 1)  # (3000, 240)
    x = mm(A, np.ascontiguousarray(c1w.reshape(O, C * KS).T)) + W["w.conv1.bias"]
    x = _timed("gelu", gelu, x)                  # (3000, 384)

    c2w = W["w.conv2.weight"]                    # (384, 384, 3)
    O2, C2, KS2 = c2w.shape
    A = _timed("im2col", im2col, np.ascontiguousarray(x.T), KS2, 2, 1)  # (1500, 1152)
    x = mm(A, np.ascontiguousarray(c2w.reshape(O2, C2 * KS2).T)) + W["w.conv2.bias"]
    x = _timed("gelu", gelu, x)                  # (1500, 384)

    # Sliced, not assumed equal: a shortened mel gives fewer than 1500
    # frames and the positional embedding has to be cut with it.
    x = x + W["w.positional_embedding"][:x.shape[0]]
    ctx, d = x.shape
    hd = d // n_head
    scale = hd ** -0.25

    for L in range(n_layer):
        p = f"w.blocks.{L}."
        h = _timed("layernorm", layer_norm, x, W[p + "attn_ln.weight"],
                   W[p + "attn_ln.bias"])
        # akey: q/k/v are three GEMMs of one design over the SAME activation, so
        # the backend converts and uploads it once instead of three times.
        # bias folded into the GEMM, not added afterwards: q/k/v/out then share
        # ONE design with fc1, which is what keeps the encoder inside six
        # hw_contexts once fc1 needs the fold for its fused GELU. Whisper's key
        # projection has no bias, so it folds a zero row.
        ak = f"L{L}.attn_ln"
        if fold_bias:
            q = mm(h, W[p + "attn.query.weight_T"], akey=ak,
                   bias=W[p + "attn.query.bias"])
            k = mm(h, W[p + "attn.key.weight_T"], akey=ak, bias=zero_bias(d))
            v = mm(h, W[p + "attn.value.weight_T"], akey=ak,
                   bias=W[p + "attn.value.bias"])
        else:
            q = mm(h, W[p + "attn.query.weight_T"], akey=ak) + \
                W[p + "attn.query.bias"]
            k = mm(h, W[p + "attn.key.weight_T"], akey=ak)
            v = mm(h, W[p + "attn.value.weight_T"], akey=ak) + \
                W[p + "attn.value.bias"]

        if attn_fn is not None:
            # QK -> softmax -> AV as one device-resident chain: the caller owns
            # the whole block, so no attention activation touches host memory.
            o = _timed("attn_chain", attn_fn, q, k, v, n_head, scale)
        else:
            o = np.empty_like(q)
            # Both attention GEMMs use a different compiled design, and
            # switching xclbins costs ~1.5 ms (probe_switch.py). Run all 6 Q@K^T
            # calls back-to-back, then all 6 W@V calls, so each layer pays 2
            # switches instead of 12.
            wgts = []
            for hh in range(n_head):
                sl = slice(hh * hd, (hh + 1) * hd)
                qh = _timed("attn_prep", np.ascontiguousarray, q[:, sl] * scale)
                kh = _timed("attn_prep", np.ascontiguousarray, k[:, sl].T) * scale
                wgts.append(attn_mm(qh, kh))                  # (ctx, ctx)
            if npu_softmax:
                wgts = npu_softmax_stack(wgts)
            else:
                wgts = [_timed("softmax", softmax, s) for s in wgts]
            for hh in range(n_head):
                sl = slice(hh * hd, (hh + 1) * hd)
                o[:, sl] = attn_mm(wgts[hh],
                                   _timed("attn_prep", np.ascontiguousarray,
                                          v[:, sl]))
        x = (x + mm(o, W[p + "attn.out.weight_T"], bias=W[p + "attn.out.bias"])
             if fold_bias else
             x + mm(o, W[p + "attn.out.weight_T"]) + W[p + "attn.out.bias"])

        h = _timed("layernorm", layer_norm, x, W[p + "mlp_ln.weight"],
                   W[p + "mlp_ln.bias"])
        parts = W.get(p + "mlp.0.parts")
        if fused_gelu:
            f = npu_matmul_gelu(h, W[p + "mlp.0.weight_T"], W[p + "mlp.0.bias"])
        elif parts is not None and mlp_fn is not None:
            # fc1 chunks -> GELU -> fc2 without the 5120-wide activation coming
            # back to host memory between the stages. The backend drives the
            # chunk GEMMs itself so it can overlap them with their own staging.
            x = x + _timed("mlp_dev", mlp_fn, h, parts,
                           W[p + "mlp.2.weight_T"],
                           f"L{L}.mlp_ln") + W[p + "mlp.2.bias"]
            continue
        elif parts is not None:
            ak = f"L{L}.mlp_ln"
            f = _timed("gelu", gelu, _tap_fc1(np.concatenate(
                [mm(h, wc, akey=ak) + bc for wc, bc in parts], axis=1)))
        else:
            f = _timed("gelu", gelu, _tap_fc1(
                mm(h, W[p + "mlp.0.weight_T"]) + W[p + "mlp.0.bias"]))
        x = x + mm(f, W[p + "mlp.2.weight_T"]) + W[p + "mlp.2.bias"]

    x = layer_norm(x, W["w.ln_post.weight"], W["w.ln_post.bias"])
    if prof is not None:
        prof["total"] = tic() - t_all
    return x


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["numpy", "npu"], default="npu")
    ap.add_argument("--attn", choices=["cpu", "npu"], default="npu")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--ref", type=str, default=REF)
    ap.add_argument("--npu-softmax", action="store_true",
                    help="attention softmax on the NPU (correct row-wise kernel)")
    ap.add_argument("--fused-gelu", action="store_true",
                    help="run mlp.fc1 through the matmul design with GELU fused "
                         "into the core (bias folded into the GEMM)")
    args = ap.parse_args()

    z = np.load(args.ref)
    W = {k: z[k].astype(np.float32) for k in z.files}
    mel, ref = W.pop("mel"), W.pop("ref_out")
    # pre-transpose every Linear weight once (torch stores (out,in))
    for k in [k for k in W if k.endswith(".weight") and W[k].ndim == 2
              and not k.endswith("_ln.weight")]:
        W[k[:-len("weight")] + "weight_T"] = np.ascontiguousarray(W[k].T)

    # Read the geometry off the weights rather than trusting the tiny-shaped
    # defaults: on a large-v3 reference those would compute 4 of 32 layers and
    # return a plausible-looking wrong answer instead of failing.
    n_layer = 1 + max(int(k.split(".")[2]) for k in W if k.startswith("w.blocks."))
    n_head = W["w.ln_post.weight"].shape[0] // 64
    print(f"geometry from weights: n_layer={n_layer} n_head={n_head}")

    if args.backend == "npu":
        npu_init()
        mm = npu_matmul
        attn_mm = npu_matmul if args.attn == "npu" else np_matmul
    else:
        mm = np_matmul
        attn_mm = np_matmul

    prof = {}
    for i in range(args.iters):
        CPU_PROF.clear()
        _npu["calls"] = 0
        _npu["npu_ns"] = 0
        _npu["e2e"] = 0.0
        _npu["shapes"] = {}
        out = encoder(W, mel, mm, attn_mm, n_layer=n_layer, n_head=n_head,
                      prof=prof, fused_gelu=args.fused_gelu,
                      npu_softmax=args.npu_softmax)
        gemm_e2e = _npu.get("e2e", 0.0)
        print(f"[iter {i}] backend={args.backend} attn={args.attn} "
              f"total={prof['total']*1e3:.1f} ms  "
              f"gemm_e2e={gemm_e2e*1e3:.1f} ms  "
              f"npu_kernel={_npu.get('npu_ns',0)/1e6:.1f} ms  "
              f"calls={_npu.get('calls',0)}  "
              f"non-gemm={(prof['total']-gemm_e2e)*1e3:.1f} ms")
        print("          cpu ops: " + "  ".join(
            f"{k}={v*1e3:.1f}ms" for k, v in sorted(CPU_PROF.items(),
                                                    key=lambda kv: -kv[1])))

    d = out - ref
    den = np.abs(ref).mean()
    print(f"\nvs torch float32 reference {ref.shape}:")
    print(f"  max |abs err|   : {np.abs(d).max():.6f}")
    print(f"  mean |abs err|  : {np.abs(d).mean():.6f}")
    print(f"  mean |ref|      : {den:.6f}")
    print(f"  rel L2 error    : {np.linalg.norm(d)/np.linalg.norm(ref):.6f}")
    print(f"  cosine sim      : "
          f"{float((out*ref).sum()/(np.linalg.norm(out)*np.linalg.norm(ref))):.8f}")

    if _npu.get("shapes"):
        print("\nper-shape NPU breakdown (M,K,N padded):")
        for kshape, (cnt, e2e, ns) in sorted(_npu["shapes"].items(),
                                             key=lambda kv: -kv[1][1]):
            print(f"  {kshape}  x{cnt:3d}  e2e={e2e*1e3:8.1f} ms  "
                  f"kernel={ns/1e6:8.1f} ms  ({e2e/cnt*1e6:.0f} us/call)")

    if args.save:
        np.save(args.save, out.astype(np.float32))
        print("saved", args.save)


if __name__ == "__main__":
    main()
