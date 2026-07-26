#!/usr/bin/env python3
"""One large-v3 encoder block straight through on raw XRT, with REAL data.

Unlike a launch-cost microbenchmark on zero buffers, this runs the
actual block dataflow: layernorm, q/k/v, per-head attention with its slicing and
transposes, softmax, output projection, MLP, residuals. Every GEMM goes through
the raw-XRT backend; weights are uploaded once and stay resident.

The point is the routing cost we had not been paying.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import os
import time

import numpy as np

import npu_whisper_encoder as E
import rawxrt

MAX_N = 2048


def block(x, W, nh, rt, parts, npu_gelu, chain=False):
    d = x.shape[1]
    hd = d // nh
    scale = hd ** -0.25
    mm = rt.matmul
    h = E.layer_norm(x, W["ln1_g"], W["ln1_b"])
    q = mm(h, W["Wq"]) + W["bq"]
    k = mm(h, W["Wk"])
    v = mm(h, W["Wv"]) + W["bv"]
    if chain:
        o = E._timed("attn_chain", rt.attn_chain, q, k, v, nh, scale)
    else:
        o = np.empty_like(q)
        wg = []
        for i in range(nh):
            sl = slice(i * hd, (i + 1) * hd)
            qh = E._timed("route_head", np.ascontiguousarray, q[:, sl] * scale)
            kh = E._timed("route_head", np.ascontiguousarray, k[:, sl].T) * scale
            wg.append(mm(qh, kh))
        wg = [E._timed("softmax", E.softmax, s) for s in wg]
        for i in range(nh):
            sl = slice(i * hd, (i + 1) * hd)
            vh = E._timed("route_head", np.ascontiguousarray, v[:, sl])
            o[:, sl] = mm(wg[i], vh)
    x = x + mm(o, W["Wo"]) + W["bo"]
    h = E.layer_norm(x, W["ln2_g"], W["ln2_b"])
    f32 = np.concatenate([mm(h, Wc) + bc for Wc, bc in parts], axis=1)
    f = rt.gelu(f32) if npu_gelu else E._timed("gelu", E.gelu, f32)
    x = x + mm(f, W["W2"]) + W["b2"]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=1280)
    ap.add_argument("--heads", type=int, default=20)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--npu-gelu", action="store_true")
    ap.add_argument("--chain", action="store_true",
                    help="QK -> softmax -> AV on device through shared BOs")
    a = ap.parse_args()

    d, nh, ctx = a.dim, a.heads, a.ctx
    rng = np.random.default_rng(0)
    s = 0.02
    W = {"ln1_g": np.ones(d, np.float32), "ln1_b": np.zeros(d, np.float32),
         "ln2_g": np.ones(d, np.float32), "ln2_b": np.zeros(d, np.float32),
         "bq": np.zeros(d, np.float32), "bv": np.zeros(d, np.float32),
         "bo": np.zeros(d, np.float32), "b2": np.zeros(d, np.float32)}
    for n in ("Wq", "Wk", "Wv", "Wo"):
        W[n] = rng.normal(0, s, (d, d)).astype(np.float32)
    W["W1"] = rng.normal(0, s, (d, 4 * d)).astype(np.float32)
    W["b1"] = np.zeros(4 * d, np.float32)
    W["W2"] = rng.normal(0, s, (4 * d, d)).astype(np.float32)
    x = rng.normal(0, 1.0, (ctx, d)).astype(np.float32)

    step = 1280
    parts = [(np.ascontiguousarray(W["W1"][:, j:j + step]),
              np.ascontiguousarray(W["b1"][j:j + step]))
             for j in range(0, 4 * d, step)]

    rt = rawxrt.RawRT()
    rt.register_weights([v for v in W.values()] +
                        [p for pr in parts for p in pr])

    print(f"=== one block d={d} heads={nh} ctx={ctx}  (real dataflow)")
    block(x, W, nh, rt, parts, a.npu_gelu, a.chain)   # warm / compile

    walls = []
    for i in range(a.iters):
        rawxrt.PROF.clear()
        E.CPU_PROF.clear()
        rt.calls = 0
        rt.ctx_creates = 0
        c0 = os.times()
        t0 = time.perf_counter()
        block(x, W, nh, rt, parts, a.npu_gelu, a.chain)
        wall = time.perf_counter() - t0
        c1 = os.times()
        cpu = (c1.user - c0.user) + (c1.system - c0.system)
        walls.append(wall)
        print(f"[iter {i}] wall={wall*1e3:7.1f} ms  cpu={cpu*1e3:6.1f} ms "
              f"-> {cpu/wall:.2f} cores  gemms={rt.calls} ctx={rt.ctx_creates}")
        print("          xrt: " + "  ".join(
            f"{k}={v*1e3:.1f}" for k, v in
            sorted(rawxrt.PROF.items(), key=lambda kv: -kv[1])))
        print("          np : " + "  ".join(
            f"{k}={v*1e3:.1f}" for k, v in
            sorted(E.CPU_PROF.items(), key=lambda kv: -kv[1])))
    if rt.fallbacks:
        print("\nFALLBACKS TAKEN:")
        for f in rt.fallbacks:
            print("  " + f)
    med = float(np.median(walls))
    print(f"\nblock median {med*1e3:.1f} ms  ->  x{a.layers} layers = "
          f"{med*a.layers:.2f} s")


if __name__ == "__main__":
    main()
