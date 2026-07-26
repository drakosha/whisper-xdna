#!/usr/bin/env python3
"""Correctness + timing for the row-wise NPU layernorm and softmax."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron.device import from_name
from aie.utils.hostruntime import set_current_device
from aie.utils.benchmark import run_iters

import rowops


def run(design, a_np, rows, cols, iters):
    a_t = iron.tensor(a_np, dtype=bfloat16, device="npu")
    c_t = iron.zeros((rows, cols), dtype=bfloat16, device="npu")
    b = run_iters(design, a_t, c_t, warmup=3, iters=iters, rows=rows, cols=cols)
    return c_t.numpy().reshape(rows, cols).astype(np.float32), b


def main():
    bad = []           # every check that failed, so the exit code can say so

    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    set_current_device(from_name("npu", n_cols=4))
    rng = np.random.default_rng(0)

    print("=== softmax (correctness is the point) ===")
    for rows, cols, label in [(512 * 20, 512, "L=512, 20 heads stacked"),
                              (1536 * 6, 1536, "L=1536, 6 heads stacked")]:
        # realistic attention logits: whisper scales q and k by (d_head)^-0.25
        # each, so scores land in roughly this range
        a_np = rng.normal(0, 6.0, size=(rows, cols)).astype(bfloat16)
        out, b = run(rowops.softmax_row, a_np, rows, cols, args.iters)
        x = a_np.astype(np.float32)
        e = np.exp(x - x.max(axis=1, keepdims=True))
        ref = e / e.sum(axis=1, keepdims=True)
        rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
        rowsum = out.sum(axis=1)
        print(f"  {label}: {rows}x{cols}  npu min={b.npu.min_us/1e3:.3f} ms "
              f"avg={b.npu.avg_us/1e3:.3f} ms")
        print(f"    rel L2 vs fp32 = {rel:.6f}   row sums: min={rowsum.min():.5f} "
              f"max={rowsum.max():.5f}   finite={np.isfinite(out).all()}")
        ok = rel < 0.02 and np.isfinite(out).all()
        bad += [] if ok else [f"softmax {label} {rows}x{cols}"]
        print("    PASS" if ok else "    FAIL")

    # extreme logits: the stock kernel has no max subtraction and overflows here
    print("  overflow check (logits up to +200):")
    rows, cols = 512, 512
    a_np = (rng.normal(0, 6.0, size=(rows, cols)) + 200.0).astype(bfloat16)
    out, _ = run(rowops.softmax_row, a_np, rows, cols, 3)
    x = a_np.astype(np.float32)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    ref = e / e.sum(axis=1, keepdims=True)
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    print(f"    rel L2 = {rel:.6f}  finite={np.isfinite(out).all()}  "
          f"row sum={out.sum(axis=1).mean():.5f}")

    print("=== layernorm ===")
    for rows, cols in [(512, 1280), (1536, 384)]:
        a_np = rng.normal(0, 1.0, size=(rows, cols)).astype(bfloat16)
        out, b = run(rowops.layer_norm, a_np, rows, cols, args.iters)
        x = a_np.astype(np.float32)
        ref = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-5)
        rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
        bad += [] if rel < 0.02 else [f"layernorm {rows}x{cols}"]
        print(f"  {rows}x{cols}: npu min={b.npu.min_us/1e3:.3f} ms  rel L2={rel:.6f}"
              + ("  PASS" if rel < 0.02 else "  FAIL"))

    if bad:
        print("FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
