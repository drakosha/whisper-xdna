#!/usr/bin/env python3
"""Raw-XRT matmul backend: a drop-in for npu_whisper_encoder.npu_matmul.

Why: the same 512x512x512 GEMM costs 1.385 ms through the IRON python wrapper
and 0.267 ms driven straight from pyxrt. This bypasses IRON at run time and
keeps compiled overlays, instruction streams, weight buffers and hw_contexts
alive across calls.

Constraints this has to live with, both measured:
  * NPU1 allows only 6 concurrent hw_contexts -- the 7th fails with
    DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22). So contexts are
    managed LRU with a ceiling below that.
  * xclbins are compiled per (M, K, N, tiles), so every distinct GEMM shape in
    the encoder is its own overlay.

On top of that it runs the attention chain QK -> softmax -> AV entirely on the
device: the QK GEMM writes bf16 straight into a sub-BO of the softmax input
buffer, softmax does all heads in one launch, and the AV GEMM reads sub-BOs of
the softmax output. Nothing between the three crosses the host boundary. See
attn_chain().
"""
import os
import subprocess
import time

import numpy as np
from ml_dtypes import bfloat16
import pyxrt

# Repo-relative, so a checkout runs wherever it is cloned. The MLIR-AIE install
# is environment-specific; override with MLIR_AIE_DIR / PEANO_INSTALL_DIR when it
# lives somewhere other than the container's /opt/mlir-aie.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Compiled overlays land in the checkout root, not next to the sources: compose
# mounts ./aot from the host so a rebuild is seconds instead of minutes.
AOT = os.environ.get("NPU_AOT_DIR") or os.path.join(ROOT, "aot")
MLIR_AIE = os.environ.get("MLIR_AIE_DIR", "/opt/mlir-aie")
WA = os.path.join(MLIR_AIE, "programming_examples/basic/matrix_multiplication/"
                  "whole_array/whole_array.py")
IRONENV = os.path.join(MLIR_AIE, "ironenv/bin/python3")
PEANO = os.environ.get(
    "PEANO_INSTALL_DIR",
    os.path.join(MLIR_AIE, "ironenv/lib/python3.13/site-packages/llvm-aie"))

MAX_CTX = 6          # the measured ceiling; the 7th context fails outright
SYNC_TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
SYNC_FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
COMPLETED = pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED

# A wedged NPU must not hang the encoder: every chained launch is waited on
# with a deadline, and blowing it trips the fallback like any other error.
RUN_TIMEOUT_MS = int(os.environ.get("NPU_RUN_TIMEOUT_MS", "10000"))

# Run host work inside the device window (submit -> host work -> wait).
# NPU_OVERLAP=0 restores the strictly serial order, for A/B only.
OVERLAP = os.environ.get("NPU_OVERLAP", "1") != "0"

# Fused attention: QK -> softmax -> PV inside the compute tiles, one overlay
# instead of three. Off by default until it is signed off.
FUSED_ATTN = os.environ.get("NPU_FUSED_ATTN", "0") != "0"
FA_BQ = int(os.environ.get("NPU_FA_BQ", "48"))
FA_BK = int(os.environ.get("NPU_FA_BK", "32"))
# The tail mask stops padded keys taking probability mass. Whether it is needed
# at all is an empirical question -- exp(0 - rowmax) may already be negligible --
# and the masked design currently mis-computes, so it is switchable.
FA_MASK = os.environ.get("NPU_FA_MASK", "0") != "0"

PROF = {}


def _t(name, dt):
    PROF[name] = PROF.get(name, 0.0) + dt


def _pad_to(v, mult):
    return ((v + mult - 1) // mult) * mult


def tiles_for(M, K, N):
    if K <= 64:
        return 32, 64, 64
    if N <= 64:
        return 64, 64, 16
    return 64, 64, 32


def padded_shape(M, K, N):
    TM, TK, TN = tiles_for(M, K, N)
    return (_pad_to(M, TM * 4 * 2), _pad_to(K, TK), _pad_to(N, TN * 4),
            TM, TK, TN)


def compile_shape(Mp, Kp, Np, TM, TK, TN, verbose=True, dtype_out="f32",
                  dtype_in="bf16"):
    """AOT-compile one overlay if it is not already on disk.

    dtype_out="bf16" is what makes a GEMM chainable: its result can be handed
    straight to the row-wise kernels, which read bf16.
    """
    pre = "mm" if dtype_out == "f32" else "mmbf"
    if dtype_in != "bf16":
        pre = f"mm{dtype_in}{dtype_out}"
    name = f"{pre}_{Mp}x{Kp}x{Np}_{TM}_{TK}_{TN}"
    xcl = f"{AOT}/{name}.xclbin"
    ins = f"{AOT}/{name}.bin"
    if os.path.exists(xcl) and os.path.exists(ins):
        return name
    os.makedirs(AOT, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = "/usr/lib/python3/dist-packages"
    env["PEANO_INSTALL_DIR"] = PEANO
    cmd = [IRONENV, WA, "--dev", "npu", "--dtype_in", dtype_in,
           "--dtype_out", dtype_out, "-M", str(Mp), "-K", str(Kp),
           "-N", str(Np), "-m", str(TM), "-k", str(TK), "-n", str(TN),
           f"--xclbin-path={xcl}", f"--insts-path={ins}"]
    if verbose:
        print(f"  compiling {name} ...", flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       cwd=os.path.dirname(WA))
    if not (os.path.exists(xcl) and os.path.exists(ins)):
        raise RuntimeError(f"compile failed for {name}:\n"
                           f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return name


def compile_rowop(op, rows, cols, valid=0, verbose=False, out_cols=0):
    """AOT-compile a row-wise elementwise overlay if not already on disk.

    valid > 0 compiles a WINDOWED design: each core is handed the first `valid`
    columns of each `cols`-wide row, which is how a chained softmax skips most
    of the GEMM padding it would otherwise normalise over.
    """
    cores = os.environ.get("ROWOP_CORES", "16")
    name = (f"{op}_{rows}x{cols}" + (f"o{out_cols}" if out_cols else "")
            + (f"v{valid}" if valid else "")
            + ("" if cores == "8" else f"c{cores}"))
    xcl, ins = f"{AOT}/{name}.xclbin", f"{AOT}/{name}.bin"
    if os.path.exists(xcl) and os.path.exists(ins):
        return name
    env = dict(os.environ)
    env["PYTHONPATH"] = "/usr/lib/python3/dist-packages"
    env["PEANO_INSTALL_DIR"] = PEANO
    r = subprocess.run(
        [IRONENV, os.path.join(HERE, "compile_rowop.py"),
         op, str(rows), str(cols), AOT, str(valid)]
        + ([str(out_cols)] if out_cols else []),
        env=env, capture_output=True, text=True,
        cwd=HERE)
    if not (os.path.exists(xcl) and os.path.exists(ins)):
        raise RuntimeError(f"rowop compile failed {name}:\n"
                           f"{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return name


def np_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


class _Entry:
    __slots__ = ("name", "ctx", "kernel", "ibo", "n_insts", "A", "B", "C",
                 "Mp", "Kp", "Np", "last", "a_key")


class _Chain:
    """Device buffers for one attention geometry (n_head, ctx, head_dim)."""
    __slots__ = ("qk", "sm", "av", "Mp", "Np", "hd", "nh", "S_in", "S_out",
                 "sub_in", "sub_out", "qbo", "kbo", "vbo", "obo",
                 "sub_q", "sub_k", "sub_v", "sub_o", "qs", "ks", "vs",
                 "rl_qk", "rl_av", "runs", "own_qk", "own_av")


class RawRT:
    def __init__(self, max_ctx=MAX_CTX):
        self.dev = pyxrt.device(0)
        self.max_ctx = max_ctx
        self._uuid = {}       # name -> (uuid, insts ndarray)
        self._live = {}       # name -> _Entry
        self._wbo = {}        # (bkey, name) -> bo holding an uploaded weight
        self._scratch = {}
        # scratch key -> design names its buffers are carved out of, so
        # eviction can drop exactly the entries that just lost their parent
        self._scratch_owner = {}
        self._known = {}      # id(weight array) -> stable key
        self._keep = []       # hold refs so those ids cannot be recycled
        self._chains = {}     # (nh, ctx, hd) -> _Chain
        self._views = {}      # id(bo) -> numpy view onto its memory
        self._clock = 0
        self.calls = 0
        self.ctx_creates = 0
        self.evictions = 0
        self.chain_ok = True  # cleared for good the first time the chain fails
        self.fused_ok = True  # the fused attention overlay, retired separately
        self.mlp_ok = True
        self.softmax_ok = True
        self.fallbacks = []   # human-readable log of every degradation

    # -- overlay / context management ------------------------------------
    def _register(self, name):
        if name in self._uuid:
            return self._uuid[name]
        xb = pyxrt.xclbin(f"{AOT}/{name}.xclbin")
        uuid = self.dev.register_xclbin(xb)
        insts = np.fromfile(f"{AOT}/{name}.bin", dtype=np.uint32)
        kname = [k.get_name() for k in xb.get_kernels()][0]
        self._uuid[name] = (uuid, insts, kname)
        return self._uuid[name]

    def _evict_lru(self):
        # Least-recently-used, but a design holding resident weights loses only
        # when nothing else is available. Plain LRU picked the fc2 design once
        # per pass -- it is the last thing a layer touches, so it ages while the
        # conv frontend of the NEXT pass opens two contexts -- and taking it out
        # dropped all 32 of its cached weights, putting 419 MB back on the bus
        # every pass. The conv, QK, softmax and AV designs
        # cache no weights at all, so they are the right victims.
        held = {}
        for bkey, name in self._wbo:
            held[name] = held.get(name, 0) + 1

        victim = min(self._live.values(),
                     key=lambda e: (held.get(e.name, 0) > 0, e.last))
        for k in [k for k, v in self._wbo.items() if k[1] == victim.name]:
            self._views.pop(id(self._wbo[k]), None)
            del self._wbo[k]
        for bo in (victim.A, victim.B, victim.C):
            if bo is not None:
                self._views.pop(id(bo), None)
        # A cached runlist keeps its hw_context alive through the kernel and run
        # objects inside it. Dropping the entry from _live is then not enough --
        # the context stays open, the ceiling of 6 is reached anyway and the next
        # create fails with DRM_IOCTL_AMDXDNA_CREATE_HWCTX (err=-22). So every
        # reference held outside _live has to go with the entry.
        for c in self._chains.values():
            if c.own_qk is victim or c.own_av is victim:
                c.rl_qk = c.rl_av = None
                c.own_qk = c.own_av = None
                c.runs = []
        # Scratch carved out of a victim's buffers has to go with it. The MLP
        # sub-BOs are bands of fc2's A operand: they survive their parent being
        # freed, the next pass allocates a new A, and fc1 then writes into
        # nothing -- normal timings, silently wrong text. The next call
        # reallocates whatever it finds missing, so dropping is always safe.
        for key in [k for k, owners in self._scratch_owner.items()
                    if victim.name in owners]:
            self._scratch.pop(key, None)
            del self._scratch_owner[key]
        del self._live[victim.name]
        self.evictions += 1

    def _ctx_entry(self, name):
        """Context + kernel + instruction BO, LRU-managed. Operand buffers are
        allocated by the caller that needs them (a chained design brings its
        own), so this is the one place that talks to the 6-context ceiling."""
        e = self._live.get(name)
        if e is not None:
            self._clock += 1
            e.last = self._clock
            return e
        while len(self._live) >= self.max_ctx:
            self._evict_lru()
        uuid, insts, kname = self._register(name)
        t0 = time.perf_counter()
        e = _Entry()
        e.name = name
        e.A = e.B = e.C = None
        e.a_key = None
        e.Mp = e.Kp = e.Np = 0
        e.ctx = pyxrt.hw_context(self.dev, uuid)
        e.kernel = pyxrt.kernel(e.ctx, kname)
        e.n_insts = insts.size
        e.ibo = pyxrt.bo(self.dev, insts.nbytes, pyxrt.bo.cacheable,
                         e.kernel.group_id(1))
        e.ibo.write(insts.tobytes(), 0)
        e.ibo.sync(SYNC_TO)
        self._clock += 1
        e.last = self._clock
        self._live[name] = e
        self.ctx_creates += 1
        _t("ctx_create", time.perf_counter() - t0)
        return e

    def _entry(self, name, Mp, Kp, Np):
        e = self._ctx_entry(name)
        if e.A is None:
            e.Mp, e.Kp, e.Np = Mp, Kp, Np
            e.A = pyxrt.bo(self.dev, Mp * Kp * 2, pyxrt.bo.host_only,
                           e.kernel.group_id(3))
            e.B = pyxrt.bo(self.dev, Kp * Np * 2, pyxrt.bo.host_only,
                           e.kernel.group_id(4))
            e.C = pyxrt.bo(self.dev, Mp * Np * 4, pyxrt.bo.host_only,
                           e.kernel.group_id(5))
        return e

    def _rowentry(self, name, n_elems):
        e = self._ctx_entry(name)
        if e.A is None:
            e.Mp = n_elems
            e.A = pyxrt.bo(self.dev, n_elems * 2, pyxrt.bo.host_only,
                           e.kernel.group_id(3))
            e.C = pyxrt.bo(self.dev, n_elems * 2, pyxrt.bo.host_only,
                           e.kernel.group_id(4))
        return e

    def register_weights(self, arrays):
        """Name arrays that never change so their device buffer is uploaded once.

        Keyed by id(), which is only safe because the references are held here
        for the lifetime of the runtime -- otherwise a freed temporary could be
        allocated at the same address and silently reuse the wrong weights.
        """
        for i, a in enumerate(arrays):
            self._known[id(a)] = f"w{i}"
            self._keep.append(a)

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _await(r, what):
        st = r.wait(RUN_TIMEOUT_MS)
        if st != COMPLETED:
            raise RuntimeError(f"{what}: run ended in {st}")
        return st

    def view(self, bo, shape, dtype=bfloat16):
        """numpy view onto a BO's own memory -- no copy in either direction.

        bo.write(a.tobytes(), 0) costs TWO memcpys of the operand (tobytes
        builds a bytes object, write copies it into the buffer) on top of the
        dtype conversion that has to happen anyway. Filling this view instead
        costs the conversion alone. Measured on a 7.86 MB buffer: 0.350 ms to
        read through bo.read(), 0.064 ms through the map.

        The mapping is stable -- a view held across launches sees each new
        result -- so it is cached per buffer. The caller owns the
        aliasing: anything handed out of here is device memory that the NEXT
        launch of the same design overwrites.
        """
        got = self._views.get(id(bo))
        if got is not None and got[0].shape == shape and got[0].dtype == dtype:
            return got[0]
        v = np.frombuffer(bo.map(), dtype=dtype)
        v = v[:int(np.prod(shape))].reshape(shape)
        # the buffer is stored WITH its view: a numpy array over bo.map() does
        # not keep the bo alive by itself, and a view onto a freed buffer is a
        # use-after-free. Dropping the entry drops both, which is what eviction
        # needs in order to actually release a weight buffer.
        self._views[id(bo)] = (v, bo)
        return v

    def _pad_into(self, bo, X, rows, cols, ones_col=None, bias_row=None):
        """Convert and pad X straight into the BO, then flush it.

        ones_col / bias_row fold a bias into the GEMM itself: a constant-1
        column on A and the bias as row K of B make the product A@B + bias,
        which is what a fused activation has to see. Both are written into the
        padded buffer that has to exist anyway, so no augmented copy of either
        operand is ever built. Exact, and it costs K 1280 -> 1344.
        """
        dst = self.view(bo, (rows, cols))
        if X.shape != (rows, cols):
            dst[X.shape[0]:, :] = 0
            dst[:X.shape[0], X.shape[1]:] = 0
        dst[:X.shape[0], :X.shape[1]] = X
        if ones_col is not None:
            dst[:X.shape[0], ones_col] = 1.0
        if bias_row is not None:
            idx, vec = bias_row
            dst[idx, :vec.shape[0]] = vec
        bo.sync(SYNC_TO)

    def _pad(self, X, rows, cols, tag):
        buf = self._scratch.get((rows, cols, tag))
        if buf is None:
            buf = np.zeros((rows, cols), dtype=bfloat16)
            self._scratch[(rows, cols, tag)] = buf
        else:
            buf[:] = 0
        buf[:X.shape[0], :X.shape[1]] = (
            X if X.dtype == bfloat16 else X.astype(bfloat16))
        return buf

    def rowop(self, op, buf_bf16, rows, cols):
        """Run a row-wise elementwise overlay over an (rows, cols) bf16 buffer."""
        name = compile_rowop(op, rows, cols)
        e = self._rowentry(name, rows * cols)
        t0 = time.perf_counter()
        self.view(e.A, (rows, cols))[:] = buf_bf16
        e.A.sync(SYNC_TO)
        _t(f"{op}_upload", time.perf_counter() - t0)
        t0 = time.perf_counter()
        r = e.kernel(3, e.ibo, e.n_insts, e.A, e.C)
        r.wait()
        _t(f"{op}_run", time.perf_counter() - t0)
        t0 = time.perf_counter()
        e.C.sync(SYNC_FROM)
        out = self.view(e.C, (rows, cols))
        _t(f"{op}_download", time.perf_counter() - t0)
        return out

    # ---- elementwise ops built on rowop --------------------------------
    # ONE shape for every GELU, so the whole encoder needs a single overlay
    # against a ceiling of 6 hw_contexts. 7680x1024 = 7864320 is exactly the
    # padded fc1 result of large-v3 (1536 x 5120), which is what lets
    # gelu_then_matmul write into fc2's A buffer without a second overlay.
    # The two conv-frontend GELUs are smaller and pad into it (~19 ms of
    # wasted lanes over the whole encoder); the row count divides by 16 cores.
    GELU_ROWS, GELU_COLS = 7680, 1024
    # "gl" = stock tanh kernel (unusable at depth, cosine 0.951 on large-v3),
    # "g2" = logistic form (gelu_row2.cc, 0.991): pays through the exp LUT,
    # "g3" = erf-GELU by piecewise polynomial on raw s (gelu_row3.cc, 0.993):
    #        5 pieces x degree 4, and every Horner step carries the state and the
    #        coefficients as bf16 hi/lo pairs to survive the conditioning.
    # "g4" = the same erf-GELU with each piece mapped onto u in [-1, 1]
    #        (gelu_row4.cc) -- the default. Centring makes plain bf16 state and
    #        plain bf16 coefficients hold, and 3 pieces then match 5, so it is 12
    #        Horner steps of {convert, multiply, MAC} against 20 of {2 converts,
    #        subtract, 6 multiplies, 3 adds}. 4878 ms against g3's 5370 at cosine
    #        parity: per sample 0.99473/0.99372/0.99186 against g3's
    #        0.99322/0.99415/0.99336 -- better on one, worse on two, mean within
    #        0.00014, all three transcriptions MATCH. The speed is the win here;
    #        the accuracy is a wash, not an improvement.
    # "g4r" = g4 plus aie::set_rounding(conv_even). The part rounds by
    #        TRUNCATION toward zero by default, which g2/g3/g4 all inherited
    #        silently. One line moved the kernel's bias from -2.0e-04 to
    #        +4.3e-05, raw L2 0.00408 -> 0.00271 and downstream 0.00857 ->
    #        0.00635 on real fc1 activations, and end-to-end cosine 0.99411 ->
    #        0.99462 against a bf16 floor of 0.99493 -- at 3674 ms against 3671,
    #        i.e. free. The tiny golden is unaffected (it runs numpy GELU), so
    #        this needs no new reference. GELU_OP=g4 restores the old rounding.
    #        NB the same line is POISON in softmax: through the exp LUT it gives
    #        cosine 0.879.
    GELU_OP = os.environ.get("GELU_OP", "g4r")

    def mlp(self, h, parts, B, akey=None, bkey=None):
        """fc2(GELU(fc1(h))) on the device, falling back to the host path.

        The whole MLP is driven from here rather than from the encoder so the
        staging of one fc1 column chunk can run while the NEXT chunk's GEMM is
        in flight -- that work is otherwise 230 ms of host time with the device
        idle behind it.
        """
        if self.mlp_ok:
            try:
                return self._mlp_dev(h, parts, B, akey, bkey)
            except Exception as exc:
                self.mlp_ok = False
                msg = (f"device mlp -> host path: "
                       f"{type(exc).__name__}: {exc}")
                self.fallbacks.append(msg)
                print(f"[rawxrt] {msg}", flush=True)
        f = self.gelu(np.concatenate(
            [self.matmul(h, wc, akey=akey) + bc for wc, bc in parts], axis=1))
        return self.matmul(f, B, bkey=bkey)

    GELU_F32_OP = os.environ.get("GELU_F32_OP", "g4f")

    def _mlp_dev(self, h, parts, B, akey=None, bkey=None):
        """fc2( GELU(fc1(h)) ) with nothing in the MLP touching host memory.

        fc1's bias is folded into its own GEMM (ones column on A, bias as row K
        of B), so each column chunk writes an fp32 accumulator that is ALREADY
        the value GELU has to see. Each chunk gets its own C buffer, so all four
        are submitted before any of them is waited on; then four GELU launches
        read those accumulators and write bf16 straight into the column band
        each occupies in fc2's A operand. fc2 then runs from that buffer.

        What this removes against the previous version: the 241 ms of host
        f32->bf16 conversion (mlp_stage), the readback of every fc1 chunk, and
        the host-side bias add. It costs K 1280 -> 1344 on the folded design,
        which q/k/v/out share so the whole encoder still needs six contexts.
        """
        M = h.shape[0]
        K = sum(wc.shape[1] for wc, _ in parts)
        Mp, Kp, Np, TM, TK, TN = padded_shape(M, K, B.shape[1])
        assert Kp == K, (Kp, K)      # a padded K would need GELU(0)=0 columns
        e2 = self._entry(compile_shape(Mp, Kp, Np, TM, TK, TN, verbose=False),
                         Mp, Kp, Np)

        w1, b1 = parts[0]
        Mp1, Kp1, Np1, T1, T2, T3 = padded_shape(M, w1.shape[0] + 1, w1.shape[1])
        e1 = self._entry(compile_shape(Mp1, Kp1, Np1, T1, T2, T3, verbose=False),
                         Mp1, Kp1, Np1)
        if (Mp1, Np1) != (Mp, Np1):
            raise ValueError(f"fc1 rows {Mp1} do not match fc2 rows {Mp}")
        gname = compile_rowop(self.GELU_F32_OP, Mp, Np1, out_cols=Kp)
        ge = self._ctx_entry(gname)

        key = ("mlpdev", Mp, Kp, Np1)
        got = self._scratch.get(key)
        if got is None:
            cbufs = [pyxrt.bo(self.dev, Mp1 * Np1 * 4, pyxrt.bo.host_only,
                              e1.kernel.group_id(5)) for _ in parts]
            # a band of fc2's A operand per chunk; the design writes Np1 of
            # every Kp columns, the offset picks which band
            subs = [pyxrt.bo(e2.A, Mp * Kp * 2 - j * Np1 * 2, j * Np1 * 2)
                    for j in range(len(parts))]
            self._scratch[key] = got = (cbufs, subs)
            # subs are bands of e2.A and cbufs are sized against e1's group id:
            # both die with their design.
            self._scratch_owner[key] = {e1.name, e2.name}
        cbufs, subs = got

        t0 = time.perf_counter()
        toks = [self.mm_start(h, wc, akey=akey, bias=bc, cbo=cbufs[j])
                for j, (wc, bc) in enumerate(parts)]
        for t in toks:
            self._await(t[1], "mlp_fc1")
        _t("run", time.perf_counter() - t0)

        t0 = time.perf_counter()
        runs = [ge.kernel(3, ge.ibo, ge.n_insts, cbufs[j], subs[j])
                for j in range(len(parts))]
        for r in runs:
            self._await(r, "mlp_gelu")
        _t("mlp_gelu_run", time.perf_counter() - t0)
        # e2.A now holds a device-produced activation, not the one a_key names
        e2.a_key = None

        if bkey is None:
            bkey = self._known.get(id(B))
        wk = (bkey, e2.name)
        t0 = time.perf_counter()
        if bkey is not None and wk in self._wbo:
            bbo = self._wbo[wk]
        else:
            bbo = e2.B
            if bkey is not None:
                bbo = pyxrt.bo(self.dev, Kp * Np * 2, pyxrt.bo.host_only,
                               e2.kernel.group_id(4))
                self._wbo[wk] = bbo
            self._pad_into(bbo, B, Kp, Np)
        _t("upload_B", time.perf_counter() - t0)

        t0 = time.perf_counter()
        self._await(e2.kernel(3, e2.ibo, e2.n_insts, e2.A, bbo, e2.C), "mlp_fc2")
        _t("run", time.perf_counter() - t0)
        self.calls += 1 + len(parts)
        t0 = time.perf_counter()
        e2.C.sync(SYNC_FROM)
        out = self.view(e2.C, (Mp, Np), np.float32)[:M, :B.shape[1]].copy()
        _t("download_C", time.perf_counter() - t0)
        return out

    def gelu(self, x):
        """GELU over any shaped array: elementwise, so the buffer is reshaped
        into the single compiled (9216, 256) tile and padded on the last chunk.
        Keeps GELU to ONE overlay instead of one per tensor shape."""
        chunk = self.GELU_ROWS * self.GELU_COLS
        flat = np.ascontiguousarray(x, dtype=np.float32).ravel()
        n = flat.size
        out = np.empty(n, dtype=np.float32)
        buf = self._scratch.setdefault(
            ("gelubuf",), np.zeros(chunk, dtype=bfloat16))
        for off in range(0, n, chunk):
            take = min(chunk, n - off)
            buf[:take] = flat[off:off + take].astype(bfloat16)
            if take < chunk:
                buf[take:] = 0
            r = self.rowop(self.GELU_OP,
                           buf.reshape(self.GELU_ROWS, self.GELU_COLS),
                           self.GELU_ROWS, self.GELU_COLS)
            out[off:off + take] = r.reshape(-1)[:take].astype(np.float32)
        return out.reshape(x.shape)

    SM_OP = os.environ.get("SM_OP", "s3")   # s3 = split exp + split recip

    def softmax_stack(self, mats):
        """Row-wise softmax for every attention head of a layer in one launch."""
        nh = len(mats)
        L0, L1 = mats[0].shape
        cols = ((L1 + 15) // 16) * 16
        rows = nh * L0
        assert rows % 8 == 0, rows
        big = self._scratch.get(("smbuf", rows, cols))
        if big is None:
            big = np.empty((rows, cols), dtype=bfloat16)
            self._scratch[("smbuf", rows, cols)] = big
        big[:] = -30000.0                     # padded columns -> exp(x-max)~0
        for i, m in enumerate(mats):
            big[i * L0:(i + 1) * L0, :L1] = m.astype(bfloat16)
        r = self.rowop(self.SM_OP, big, rows, cols)
        f = r.astype(np.float32)
        return [f[i * L0:(i + 1) * L0, :L1] for i in range(nh)]

    # -- the attention chain, device resident ----------------------------
    def _chain(self, nh, ctx, hd):
        """Designs and buffers for one attention geometry.

        Layout: the six per-head QK results live in ONE parent BO as a
        (nh*Mp, Np) bf16 stack, each head a contiguous sub-BO. That is exactly
        what the row-wise softmax design wants to see, so all heads normalise in
        a single launch, and the AV GEMM then reads its A operand as a sub-BO of
        the softmax output. No activation crosses the host boundary between the
        three stages.
        """
        key = (nh, ctx, hd)
        c = self._chains.get(key)
        if c is not None:
            return c
        c = _Chain()
        # The two GEMMs of the chain want different tile shapes -- QK is skinny
        # in K, AV is skinny in N -- and each shape implies its own row padding.
        # Taking each from padded_shape() independently makes them disagree
        # whenever ctx is not a multiple of 512 (QK rounds to 256, AV to 512), and
        # the AV GEMM reads the QK output, so the chain then asserted and dropped
        # to the host path in silence: at ctx=1152 that cost 10.0 s against 5.4 s
        # at the LARGER ctx=1500. One row padding is derived for both stages.
        TMq, TKq, TNq = tiles_for(ctx, hd, ctx)
        TMa, TKa, TNa = tiles_for(ctx, ctx, hd)
        Mp = _pad_to(ctx, max(TMq, TMa) * 4 * 2)
        Np = _pad_to(ctx, max(TNq * 4, TKa))
        Kp = _pad_to(hd, TKq)
        assert Kp == hd, (Kp, hd)     # QK is one k-step: nothing accumulates
        c.Mp, c.Np = Mp, Np
        c.qk = compile_shape(Mp, Kp, Np, TMq, TKq, TNq, dtype_out="bf16")
        # The QK output row is ctx real logits padded out to Np with zeros, and
        # zeros are not -inf: a full-row softmax would hand 1.4% of the average
        # row's probability mass (8.9% on the worst row, measured) to columns
        # that are not attention. The windowed
        # design normalises over the first ceil(ctx/16)*16 columns instead, which
        # is the smallest window the 16-wide vector kernel can express.
        smop = os.environ.get("CHAIN_SM", "s4w")
        win = ((ctx + 15) // 16) * 16 if smop.endswith("w") else 0
        if win >= Np:
            # At audio_ctx=512 the GEMM does not pad the row at all: there is
            # nothing to mask, so use the plain design instead of compiling a
            # windowed one that would read the whole row anyway.
            smop, win = "s3", 0
        c.sm = compile_rowop(smop, nh * Mp, Np, valid=win)
        aN = _pad_to(hd, TNa * 4)
        c.av = compile_shape(Mp, Np, aN, TMa, TKa, TNa)

        # buffers are allocated against the softmax kernel's banks; a BO from
        # one design feeding another is legal (probe_bo_share.py)
        esm = self._ctx_entry(c.sm)
        head_b = Mp * Np * 2
        c.S_in = pyxrt.bo(self.dev, nh * head_b, pyxrt.bo.host_only,
                          esm.kernel.group_id(3))
        c.S_out = pyxrt.bo(self.dev, nh * head_b, pyxrt.bo.host_only,
                           esm.kernel.group_id(4))
        c.sub_in = [pyxrt.bo(c.S_in, head_b, h * head_b) for h in range(nh)]
        c.sub_out = [pyxrt.bo(c.S_out, head_b, h * head_b) for h in range(nh)]

        # Q/K/V and the AV output are stacked per head the same way the scores
        # are, so a whole layer's operands are one write + one sync instead of
        # 20, and every launch of a stage can be handed to the device as one
        # runlist (see _chain_runlists).
        assert aN == hd, (aN, hd)
        eqk = self._ctx_entry(c.qk)
        qh, kh = Mp * hd * 2, hd * Np * 2
        c.qbo = pyxrt.bo(self.dev, nh * qh, pyxrt.bo.host_only,
                         eqk.kernel.group_id(3))
        c.kbo = pyxrt.bo(self.dev, nh * kh, pyxrt.bo.host_only,
                         eqk.kernel.group_id(4))
        c.sub_q = [pyxrt.bo(c.qbo, qh, h * qh) for h in range(nh)]
        c.sub_k = [pyxrt.bo(c.kbo, kh, h * kh) for h in range(nh)]
        eav = self._ctx_entry(c.av)
        vh, oh = Mp * aN * 2, Mp * aN * 4
        c.vbo = pyxrt.bo(self.dev, nh * vh, pyxrt.bo.host_only,
                         eav.kernel.group_id(4))
        c.obo = pyxrt.bo(self.dev, nh * oh, pyxrt.bo.host_only,
                         eav.kernel.group_id(5))
        c.sub_v = [pyxrt.bo(c.vbo, vh, h * vh) for h in range(nh)]
        c.sub_o = [pyxrt.bo(c.obo, oh, h * oh) for h in range(nh)]
        # staging arrays; the padding tails are written once and never touched
        c.nh, c.hd = nh, hd
        c.qs = self.view(c.qbo, (nh, Mp, hd))
        c.ks = self.view(c.kbo, (nh, hd, Np))
        c.vs = self.view(c.vbo, (nh, Mp, aN))
        for a in (c.qs, c.ks, c.vs):
            a[:] = 0          # the padding tails are written once, never again
        c.rl_qk = c.rl_av = None
        c.own_qk = c.own_av = None
        c.runs = []
        self._chains[key] = c
        return c

    def _chain_runlists(self, c):
        """One pyxrt.runlist per attention stage, built once and re-executed.

        Every argument of every launch is a fixed sub-BO, so the lists are
        rebuilt only when LRU has thrown away the hw_context they were bound to
        -- a runlist belongs to one context, and the runs inside it to one
        kernel object.
        """
        eqk = self._ctx_entry(c.qk)
        eav = self._ctx_entry(c.av)
        if c.rl_qk is not None and c.own_qk is eqk and c.own_av is eav:
            return c.rl_qk, c.rl_av
        c.runs = []
        rls = []
        for e, ins in ((eqk, [(c.sub_q, c.sub_k, c.sub_in)]),
                       (eav, [(c.sub_out, c.sub_v, c.sub_o)])):
            rl = pyxrt.runlist(e.ctx)
            a, b, out = ins[0]
            for h in range(c.nh):
                r = pyxrt.run(e.kernel)
                r.set_arg(0, 3)
                r.set_arg(1, e.ibo)
                r.set_arg(2, e.n_insts)
                r.set_arg(3, a[h])
                r.set_arg(4, b[h])
                r.set_arg(5, out[h])
                rl.add(r)
                c.runs.append(r)      # keep the runs alive for the list
            rls.append(rl)
        c.rl_qk, c.rl_av = rls
        c.own_qk, c.own_av = eqk, eav
        return c.rl_qk, c.rl_av

    # -- fused attention -------------------------------------------------
    def _fchain(self, nh, ctx, hd):
        key = ("fused", nh, ctx, hd)
        c = self._chains.get(key)
        if c is not None:
            return c
        Mp = _pad_to(ctx, 16 * FA_BQ)
        assert Mp % FA_BK == 0, (Mp, FA_BK)
        name = (f"attnm_{Mp}x{hd}_q{FA_BQ}k{FA_BK}"
                + (f"v{ctx}" if FA_MASK and ctx != Mp else ""))
        e = self._ctx_entry(name)
        qb, kvb, ob = Mp * hd * 2, 2 * Mp * hd * 2, Mp * hd * 2
        c = _Chain()
        c.nh, c.hd, c.Mp, c.Np = nh, hd, Mp, hd
        c.qbo = pyxrt.bo(self.dev, nh * qb, pyxrt.bo.host_only,
                         e.kernel.group_id(3))
        c.kbo = pyxrt.bo(self.dev, nh * kvb, pyxrt.bo.host_only,
                         e.kernel.group_id(4))
        c.obo = pyxrt.bo(self.dev, nh * ob, pyxrt.bo.host_only,
                         e.kernel.group_id(5))
        c.sub_q = [pyxrt.bo(c.qbo, qb, h * qb) for h in range(nh)]
        c.sub_k = [pyxrt.bo(c.kbo, kvb, h * kvb) for h in range(nh)]
        c.sub_o = [pyxrt.bo(c.obo, ob, h * ob) for h in range(nh)]
        c.qs = self.view(c.qbo, (nh, Mp * hd))
        c.ks = self.view(c.kbo, (nh, 2 * Mp * hd))
        c.vs = self.view(c.obo, (nh, Mp * hd))
        c.qk = c.sm = c.av = name
        c.rl_qk = c.rl_av = c.own_qk = c.own_av = None
        c.runs = []
        self._chains[key] = c
        return c

    def _frunlist(self, c):
        e = self._ctx_entry(c.qk)
        if c.rl_qk is not None and c.own_qk is e:
            return c.rl_qk
        rl = pyxrt.runlist(e.ctx)
        c.runs = []
        for h in range(c.nh):
            r = pyxrt.run(e.kernel)
            for i, val in enumerate((3, e.ibo, e.n_insts, c.sub_q[h],
                                     c.sub_k[h], c.sub_o[h])):
                r.set_arg(i, val)
            rl.add(r)
            c.runs.append(r)
        c.rl_qk, c.own_qk = rl, e
        return rl

    def attn_fused(self, q, k, v, nh, scale):
        """Attention with the scores never leaving the compute tiles.

        Operands are tiled into mmul's 4x8 / 8x4 / 4x4 block layouts here, in
        ONE batched transpose per tensor -- the same shape of work the chain was
        already doing to split heads, so it replaces chain_up_qkv rather than
        adding to it. Q and K go in UNSCALED: fused_softmax applies 1/sqrt(dk).
        """
        ctx, d = q.shape
        hd = d // nh
        assert abs(scale - hd ** -0.25) < 1e-6, scale
        c = self._fchain(nh, ctx, hd)
        Mp, bq, bk = c.Mp, FA_BQ, FA_BK
        nqb, nkb = Mp // bq, Mp // bk
        rl = self._frunlist(c)

        t0 = time.perf_counter()
        qh = np.zeros((nh, Mp, hd), np.float32)
        kh = np.zeros((nh, Mp, hd), np.float32)
        vh = np.zeros((nh, Mp, hd), np.float32)
        qh[:, :ctx] = q.reshape(ctx, nh, hd).transpose(1, 0, 2)
        kh[:, :ctx] = k.reshape(ctx, nh, hd).transpose(1, 0, 2)
        vh[:, :ctx] = v.reshape(ctx, nh, hd).transpose(1, 0, 2)
        # A layout (Q, K): 4x8 sub-tiles, k-major block order
        c.qs[:] = (qh.reshape(nh, nqb, bq // 4, 4, hd // 8, 8)
                   .transpose(0, 1, 4, 2, 3, 5).reshape(nh, -1))
        kt = (kh.reshape(nh, nkb, bk // 4, 4, hd // 8, 8)
              .transpose(0, 1, 4, 2, 3, 5).reshape(nh, nkb, bk * hd))
        # B layout (V): 8x4 sub-tiles, n-major block order
        vt = (vh.reshape(nh, nkb, bk // 8, 8, hd // 4, 4)
              .transpose(0, 1, 4, 2, 3, 5).reshape(nh, nkb, bk * hd))
        c.ks[:] = np.concatenate([kt[:, :, None], vt[:, :, None]],
                                 axis=2).reshape(nh, -1)
        c.qbo.sync(SYNC_TO)
        c.kbo.sync(SYNC_TO)
        _t("fa_upload", time.perf_counter() - t0)

        t0 = time.perf_counter()
        rl.execute()
        rl.wait()
        _t("fa_run", time.perf_counter() - t0)

        t0 = time.perf_counter()
        c.obo.sync(SYNC_FROM)
        o = (c.vs.reshape(nh, nqb, hd // 4, bq // 4, 4, 4)
             .transpose(0, 1, 3, 4, 2, 5).reshape(nh, Mp, hd))
        out = np.ascontiguousarray(
            o[:, :ctx].transpose(1, 0, 2).reshape(ctx, d).astype(np.float32))
        _t("fa_download", time.perf_counter() - t0)
        self.calls += nh
        return out

    def attn_chain(self, q, k, v, nh, scale):
        """o = concat_h softmax(q_h k_h^T) v_h, computed without the activations
        ever coming back to host memory.

        Three steps down, one at a time: the fused overlay, then the chain of
        three, then the host path. A fused failure must retire ONLY the fused
        overlay -- clearing chain_ok here would skip the chain as well and cost
        the 1.7x it is worth, while the log says the chain is what took over.
        """
        if FUSED_ATTN and self.fused_ok and self.chain_ok:
            try:
                return self.attn_fused(q, k, v, nh, scale)
            except Exception as exc:
                self.fused_ok = False
                msg = f"fused attention -> chain: {type(exc).__name__}: {exc}"
                self.fallbacks.append(msg)
                print(f"[rawxrt] {msg}", flush=True)
        if self.chain_ok:
            try:
                return self._attn_chain_dev(q, k, v, nh, scale)
            except Exception as exc:
                self.chain_ok = False
                msg = f"attention chain -> host path: {type(exc).__name__}: {exc}"
                self.fallbacks.append(msg)
                print(f"[rawxrt] {msg}", flush=True)
        return self._attn_host(q, k, v, nh, scale)

    def _attn_chain_dev(self, q, k, v, nh, scale):
        ctx, d = q.shape
        hd = d // nh
        c = self._chain(nh, ctx, hd)
        Mp = c.Mp
        rl_qk, rl_av = self._chain_runlists(c)

        # one transpose per operand instead of 20 column slices: head h of a
        # (ctx, nh*hd) activation is (ctx, nh, hd)[:, h], so the whole layer is
        # staged with three conversions and three writes.
        t0 = time.perf_counter()
        c.qs[:, :ctx, :] = (q * scale).reshape(ctx, nh, hd).transpose(1, 0, 2)
        c.ks[:, :, :ctx] = (k * scale).reshape(ctx, nh, hd).transpose(1, 2, 0)
        c.qbo.sync(SYNC_TO)
        c.kbo.sync(SYNC_TO)
        _t("chain_up_qkv", time.perf_counter() - t0)

        # V is not read until the AV stage, so its transpose and upload go
        # inside the QK window instead of ahead of it.
        def stage_v():
            t0 = time.perf_counter()
            c.vs[:, :ctx, :] = v.reshape(ctx, nh, hd).transpose(1, 0, 2)
            c.vbo.sync(SYNC_TO)
            _t("chain_up_qkv", time.perf_counter() - t0)

        if not OVERLAP:
            stage_v()
        t_qk = time.perf_counter()
        rl_qk.execute()
        if OVERLAP:
            stage_v()
        rl_qk.wait()
        _t("chain_run_qk", time.perf_counter() - t_qk)

        esm = self._ctx_entry(c.sm)
        t0 = time.perf_counter()
        self._await(esm.kernel(3, esm.ibo, esm.n_insts, c.S_in, c.S_out), "sm")
        _t("chain_run_sm", time.perf_counter() - t0)

        t0 = time.perf_counter()
        rl_av.execute()
        rl_av.wait()
        _t("chain_run_av", time.perf_counter() - t0)

        t0 = time.perf_counter()
        c.obo.sync(SYNC_FROM)
        o = self.view(c.obo, (nh, Mp, hd), np.float32)
        o = np.ascontiguousarray(
            o[:, :ctx, :].transpose(1, 0, 2).reshape(ctx, d))
        _t("chain_down_av", time.perf_counter() - t0)
        self.calls += 2 * nh
        return o

    def _attn_host(self, q, k, v, nh, scale):
        """The pre-chain path: every stage round-trips through host memory."""
        ctx, d = q.shape
        hd = d // nh
        o = np.empty((ctx, d), np.float32)
        wg = []
        for h in range(nh):
            sl = slice(h * hd, (h + 1) * hd)
            qh = np.ascontiguousarray(q[:, sl] * scale)
            kh = np.ascontiguousarray(k[:, sl].T) * scale
            wg.append(self.matmul(qh, kh))
        if self.softmax_ok:
            try:
                wg = self.softmax_stack(wg)
            except Exception as exc:
                self.softmax_ok = False
                msg = (f"npu softmax -> numpy: {type(exc).__name__}: {exc}")
                self.fallbacks.append(msg)
                print(f"[rawxrt] {msg}", flush=True)
                wg = [np_softmax(s) for s in wg]
        else:
            wg = [np_softmax(s) for s in wg]
        for h in range(nh):
            sl = slice(h * hd, (h + 1) * hd)
            o[:, sl] = self.matmul(wg[h], np.ascontiguousarray(v[:, sl]))
        return o

    # -- the call --------------------------------------------------------
    # The call is split into a submit half and a collect half so a caller with
    # host work in hand can run it INSIDE the device window instead of after it.
    # The timeline was strictly serial before this: ~2.8 s of device time and
    # ~1.4 s of host time adding up to the 4.15 s wall. Nothing here changes what
    # is computed -- only when the host does its share.
    def mm_start(self, A, B, akey=None, bkey=None, bias=None, cbo=None):
        """Upload operands and submit. The run is in flight when this returns;
        the caller must mm_finish() the handle before starting another GEMM of
        the same design, unless it passed a C buffer of its own via cbo.

        bias is folded into the GEMM rather than added afterwards -- see
        _pad_into. That is what lets an activation stay on the device: the
        result written to C is already A@B + bias, so a fused GELU can read it
        without the host touching anything.
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, (A.shape, B.shape)
        if bkey is None:
            bkey = self._known.get(id(B))
        Kfold = K + 1 if bias is not None else K
        Mp, Kp, Np, TM, TK, TN = padded_shape(M, Kfold, N)
        name = compile_shape(Mp, Kp, Np, TM, TK, TN, verbose=False)
        e = self._entry(name, Mp, Kp, Np)

        # An activation that feeds several GEMMs of the SAME design -- the
        # attn_ln output going to q/k/v, or the mlp_ln output going to every fc1
        # column chunk -- is converted, padded and uploaded once. akey names the
        # activation; e.A holds whatever was uploaded last, so a mismatch (or an
        # unnamed operand) re-uploads and re-labels the buffer.
        t0 = time.perf_counter()
        akey_f = akey if bias is None else (akey, "1col")
        if akey is None or e.a_key != akey_f:
            self._pad_into(e.A, A, Mp, Kp,
                           ones_col=(K if bias is not None else None))
            e.a_key = akey_f
        _t("upload_A", time.perf_counter() - t0)

        wk = (bkey, name)
        t0 = time.perf_counter()
        if bkey is not None and wk in self._wbo:
            bbo = self._wbo[wk]
        else:
            if bkey is not None:
                bbo = pyxrt.bo(self.dev, Kp * Np * 2, pyxrt.bo.host_only,
                               e.kernel.group_id(4))
                self._wbo[wk] = bbo
            else:
                bbo = e.B
            self._pad_into(bbo, B, Kp, Np,
                           bias_row=((K, bias) if bias is not None else None))
        _t("upload_B", time.perf_counter() - t0)

        C = e.C if cbo is None else cbo
        r = e.kernel(3, e.ibo, e.n_insts, e.A, bbo, C)
        self.calls += 1
        return (e, r, M, N, Mp, Np, time.perf_counter(), C)

    def mm_finish(self, h):
        e, r, M, N, Mp, Np, t_sub, C = h
        t0 = time.perf_counter()
        self._await(r, "matmul")
        _t("run", time.perf_counter() - t0)
        # how much of each device window the caller actually filled with host
        # work -- the number this whole split exists to raise
        _t("run_overlapped", t0 - t_sub)
        t0 = time.perf_counter()
        C.sync(SYNC_FROM)
        out = self.view(C, (Mp, Np), np.float32)[:M, :N].copy()
        _t("download_C", time.perf_counter() - t0)
        return out

    def matmul(self, A, B, akey=None, bkey=None, bias=None):
        """A (M,K) f32 @ B (K,N) f32 -> (M,N) f32, bf16 in, fp32 accumulate."""
        return self.mm_finish(
            self.mm_start(A, B, akey=akey, bkey=bkey, bias=bias))
