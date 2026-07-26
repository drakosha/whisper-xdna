#!/usr/bin/env python3
"""whisper-tiny encoder end to end with every GEMM driven straight from pyxrt.

Same encoder structure that already validated at cosine 0.99991 through IRON;
only the matmul backend changes. Elementwise stays in numpy (that is the
configuration that was both fastest and most accurate).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repo_paths import ROOT
import argparse
import os
import time

import numpy as np

import npu_whisper_encoder as E
import rawxrt

REF = str(ROOT / "ref.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--ref", type=str, default=REF)
    ap.add_argument("--npu-gelu", action="store_true")
    ap.add_argument("--npu-softmax", action="store_true")
    ap.add_argument("--mel", type=str, default=None,
                    help="npz with this sample's mel + ref_out")
    ap.add_argument("--mlp-chunk", type=int, default=0,
                    help="force fc1 column chunk width (0 = decide from shape)")
    ap.add_argument("--chain", action="store_true",
                    help="QK -> softmax -> AV on device, shared BOs")
    ap.add_argument("--host-mlp", action="store_true",
                    help="keep fc1 -> GELU -> fc2 on the host path (A/B only)")
    ap.add_argument("--dump-fc1-acts", type=str, default=None,
                    metavar="PATH",
                    help="write sampled fc1 pre-activations for "
                         "tools/gen_gelu*_coeffs.py (host MLP path only)")
    ap.add_argument("--ctx", type=int, default=0,
                    help="encoder context in frames (0 = full 1500). Cuts the "
                         "mel to 2*ctx and the positional embedding with it")
    args = ap.parse_args()

    z = np.load(args.ref)
    W = {k: z[k].astype(np.float32) for k in z.files}
    mel, ref = W.pop("mel"), W.pop("ref_out")
    if args.mel:
        # weights from --ref, but this sample's mel and torch reference
        zm = np.load(args.mel)
        mel, ref = zm["mel"].astype(np.float32), zm["ref_out"].astype(np.float32)
        print(f"mel from {args.mel} {mel.shape}")
    # Read the geometry off the weights rather than taking it on flags: tiny is
    # 4 layers / 6 heads, large-v3 is 32 / 20, and a mismatch here would produce
    # a plausible-looking wrong answer instead of an error.
    n_layer = 1 + max(int(k.split(".")[2]) for k in W if k.startswith("w.blocks."))
    n_head = W["w.ln_post.weight"].shape[0] // 64
    if args.ctx:
        mel = np.ascontiguousarray(mel[:, :2 * args.ctx])
        print(f"audio_ctx={args.ctx}: mel cut to {mel.shape}")
    print(f"geometry from weights: n_layer={n_layer} n_head={n_head} "
          f"d={W['w.ln_post.weight'].shape[0]} mel={mel.shape}")
    for k in [k for k in W if k.endswith(".weight") and W[k].ndim == 2
              and not k.endswith("_ln.weight")]:
        W[k[:-len("weight")] + "weight_T"] = np.ascontiguousarray(W[k].T)

    # an f32 GEMM wider than 4096 columns does not compile (see
    # prepare_mlp_chunks); large-v3's fc1 is 5120 wide, tiny's 1536 is not
    d_model = W["w.ln_post.weight"].shape[0]
    extra = E.prepare_mlp_chunks(W, n_layer, d_model, args.mlp_chunk)
    if extra:
        print(f"fc1 split into {len(extra)//2//n_layer} column chunks per layer")

    rt = rawxrt.RawRT()
    rt.register_weights([v for v in W.values() if isinstance(v, np.ndarray)]
                        + extra)
    if args.npu_gelu:
        E.gelu = rt.gelu
    if args.npu_softmax:
        E.npu_softmax_stack = rt.softmax_stack
    kw = dict(npu_softmax=args.npu_softmax, n_layer=n_layer, n_head=n_head)
    if args.chain:
        kw["attn_fn"] = rt.attn_chain
    if args.npu_gelu and not args.host_mlp:
        kw["mlp_fn"] = rt.mlp
    # the bias fold exists so fc1 can share ONE design with q/k/v/out; without
    # the device MLP it is pure cost, and turning both off together is the A/B
    kw["fold_bias"] = not args.host_mlp

    if args.dump_fc1_acts:
        if kw.get("mlp_fn") is not None:
            sys.exit("--dump-fc1-acts needs the host MLP path (add --host-mlp): "
                     "the device MLP never brings fc1's output back to the host")
        E.FC1_TAP = []

    print("warmup (compiles every distinct overlay) ...", flush=True)
    t0 = time.perf_counter()
    out = E.encoder(W, mel, rt.matmul, rt.matmul, prof={}, **kw)
    print(f"  warmup done in {time.perf_counter()-t0:.1f} s, "
          f"{rt.calls} gemms, {rt.ctx_creates} contexts, "
          f"{rt.evictions} evictions")

    for i in range(args.iters):
        rawxrt.PROF.clear()
        E.CPU_PROF.clear()
        rt.calls = 0
        rt.ctx_creates = 0
        rt.evictions = 0
        c0 = os.times()
        t0 = time.perf_counter()
        out = E.encoder(W, mel, rt.matmul, rt.matmul, prof={}, **kw)
        wall = time.perf_counter() - t0
        c1 = os.times()
        cpu = (c1.user - c0.user) + (c1.system - c0.system)
        print(f"[iter {i}] encoder wall={wall*1e3:8.1f} ms   "
              f"cpu={cpu*1e3:7.1f} ms -> {cpu/wall:.2f} cores   "
              f"gemms={rt.calls} ctx_creates={rt.ctx_creates} "
              f"evict={rt.evictions}")
        xrtp = "  ".join(f"{k}={v*1e3:.1f}" for k, v in
                         sorted(rawxrt.PROF.items(), key=lambda kv: -kv[1]))
        npp = "  ".join(f"{k}={v*1e3:.1f}" for k, v in
                        sorted(E.CPU_PROF.items(), key=lambda kv: -kv[1]))
        print(f"          xrt: {xrtp}")
        print(f"          np : {npp}")

    if args.dump_fc1_acts:
        acts = np.concatenate(E.FC1_TAP)[:4_000_000]
        np.save(args.dump_fc1_acts, acts)
        print(f"fc1 activations: {acts.size} samples (every 97th element of "
              f"{len(E.FC1_TAP)} tensors) -> {args.dump_fc1_acts}")
        E.FC1_TAP = None

    if out.shape != ref.shape:
        # A truncated context does not reproduce the full-context reference --
        # attention over 300 keys is not attention over 1500 restricted to 300.
        # Kept only as a sanity number; transcription is the real check.
        print(f"\nreference is {ref.shape}, output {out.shape}: comparing the "
              f"overlapping prefix only")
        ref = ref[:out.shape[0]]
    d = out - ref
    print(f"\nvs torch fp32 reference {ref.shape}:")
    print(f"  rel L2 error : {np.linalg.norm(d)/np.linalg.norm(ref):.6f}")
    print(f"  cosine sim   : "
          f"{float((out*ref).sum()/(np.linalg.norm(out)*np.linalg.norm(ref))):.8f}")
    if rt.fallbacks:
        print("\nFALLBACKS TAKEN:")
        for f in rt.fallbacks:
            print("  " + f)
    if args.save:
        np.save(args.save, out.astype(np.float32))
        print("saved", args.save)


if __name__ == "__main__":
    main()
