#!/usr/bin/env python3
"""Accuracy cost of the on-device attention chain QK -> softmax -> AV.

Chaining needs QK to emit bf16 (softmax consumes bf16), which normally means
bf16 accumulation across K. For QK specifically K=64 and the tile k=64, so there
is exactly ONE k-step: nothing accumulates across steps, and the bf16 output is
just a final rounding the numpy path effectively pays too.

This simulates that by rounding the QK result to bf16 before softmax, with the
real NPU softmax, and reports the end-to-end cosine.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repo_paths import ROOT
import numpy as np
from ml_dtypes import bfloat16

import npu_whisper_encoder as E
import rawxrt

z = np.load(ROOT / "ref.npz")
W = {k: z[k].astype(np.float32) for k in z.files}
mel, ref = W.pop("mel"), W.pop("ref_out")
for k in [k for k in W if k.endswith(".weight") and W[k].ndim == 2
          and not k.endswith("_ln.weight")]:
    W[k[:-len("weight")] + "weight_T"] = np.ascontiguousarray(W[k].T)

rt = rawxrt.RawRT()
rt.register_weights([v for v in W.values() if isinstance(v, np.ndarray)])
E.npu_softmax_stack = rt.softmax_stack


def attn_mm_bf16(A, B, **kw):
    """QK as it would be with dtype_out=bf16 (what chaining requires)."""
    return rt.matmul(A, B, **kw).astype(bfloat16).astype(np.float32)


for label, amm in (("QK f32 out (today)", rt.matmul),
                   ("QK bf16 out (chainable)", attn_mm_bf16)):
    out = E.encoder(W, mel, rt.matmul, amm, prof={}, npu_softmax=True)
    d = out - ref
    cos = float((out * ref).sum() /
                (np.linalg.norm(out) * np.linalg.norm(ref)))
    print(f"{label:26s} rel L2={np.linalg.norm(d)/np.linalg.norm(ref):.6f}  "
          f"cosine={cos:.8f}  {'PASS' if cos >= 0.999 else 'FAIL'}")
    np.save(ROOT / f"out_chain_{'f32' if amm is rt.matmul else 'bf16'}.npy",
            out.astype(np.float32))
