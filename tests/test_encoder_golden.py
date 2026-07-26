#!/usr/bin/env python3
"""Golden tests for the raw-XRT whisper-tiny encoder with the device-resident
attention chain, plus the transcription integration test.

Run inside ironenv (the decode step shells out to refenv itself):

  docker compose exec npu-whisper bash -c "\\
    source /opt/mlir-aie/ironenv/bin/activate \\
    && export PYTHONPATH=/usr/lib/python3/dist-packages \\
    && export PEANO_INSTALL_DIR=/opt/mlir-aie/ironenv/lib/python3.13/site-packages/llvm-aie \\
    && python3 tests/test_encoder_golden.py"

  --update-golden   rewrite the stored golden from this run (use when a change
                    is a deliberate numerical improvement)

The stored golden is bit-exact output from one toolchain (XRT 2.21.75, MLIR-AIE
v1.3.4, firmware 1.5.5.391). On a different stack test 2 can fail while the
system is healthy -- check test 1 and test 6 first, then re-record with
--update-golden.

What each test is for:
  1 accuracy gate      cosine >= 0.999 against the torch fp32 reference
  2 golden regression  bit-identical output to the recorded run
  3 path agreement     chain and host paths must not diverge
  4 fallback           a failing chain degrades to the host path silently
  5 context ceiling    never more than MAX_CTX hw_contexts live
  6 transcription      decoded text is character-for-character the expected one
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repo_paths import ROOT
import argparse
import os
import subprocess
import time

import numpy as np

import npu_whisper_encoder as E
import rawxrt

REF = str(ROOT / "ref.npz")
GOLDEN = str(Path(__file__).resolve().parent / "data" / "golden_chain.npy")
EXPECTED = ("And so my fellow Americans ask not what your country can do for "
            "you ask what you can do for your country.")
COS_FLOOR = 0.999

_results = []


def check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
          flush=True)
    return ok


def cosine(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_weights():
    z = np.load(REF)
    W = {k: z[k].astype(np.float32) for k in z.files}
    mel, ref = W.pop("mel"), W.pop("ref_out")
    for k in [k for k in W if k.endswith(".weight") and W[k].ndim == 2
              and not k.endswith("_ln.weight")]:
        W[k[:-len("weight")] + "weight_T"] = np.ascontiguousarray(W[k].T)
    return W, mel, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-golden", action="store_true")
    args = ap.parse_args()

    W, mel, ref = load_weights()
    rt = rawxrt.RawRT()
    rt.register_weights([v for v in W.values() if isinstance(v, np.ndarray)])
    # the host path's softmax must be the raw-XRT one too, otherwise it falls
    # into the IRON wrapper and the comparison is not the comparison we mean
    E.npu_softmax_stack = rt.softmax_stack
    peak_ctx = 0

    def run(**kw):
        nonlocal peak_ctx
        t0 = time.perf_counter()
        out = E.encoder(W, mel, rt.matmul, rt.matmul, prof={}, **kw)
        peak_ctx = max(peak_ctx, len(rt._live))
        return out, time.perf_counter() - t0

    # one untimed pass: compiles, context creates and weight uploads must not
    # be charged to the numbers reported below
    run(attn_fn=rt.attn_chain)

    print("\n=== 1. chained encoder vs torch fp32 reference")
    chain, wall = run(attn_fn=rt.attn_chain)
    cos = cosine(chain, ref)
    rel = float(np.linalg.norm(chain - ref) / np.linalg.norm(ref))
    check("accuracy gate", cos >= COS_FLOOR,
          f"cosine={cos:.8f} rel L2={rel:.6f} wall={wall*1e3:.0f} ms")
    check("chain actually ran (no silent fallback)", rt.chain_ok,
          f"fallbacks={rt.fallbacks}")

    print("\n=== 2. golden regression")
    if args.update_golden or not os.path.exists(GOLDEN):
        np.save(GOLDEN, chain.astype(np.float32))
        check("golden written", True, GOLDEN)
    else:
        g = np.load(GOLDEN)
        same = np.array_equal(g, chain.astype(np.float32))
        worst = float(np.abs(g - chain).max())
        check("bit-identical to golden", same, f"max|delta|={worst:.3e}")

    print("\n=== 3. chain vs host path")
    host, hwall = run(npu_softmax=True)
    check("host path accuracy", cosine(host, ref) >= COS_FLOOR,
          f"cosine={cosine(host, ref):.8f} wall={hwall*1e3:.0f} ms")
    # The two paths are independent bf16 approximations of the same fp32
    # quantity, so they are further from each other than either is from the
    # reference -- 0.99974 and 0.99966 against torch allows them to sit at
    # ~0.9994 against one another. This gate is here to catch STRUCTURAL
    # breakage (heads swapped, a stale buffer read, padding leaking into the
    # result), which lands orders of magnitude below the floor, not to police
    # rounding.
    check("chain agrees with host path structurally",
          cosine(chain, host) >= COS_FLOOR,
          f"cosine={cosine(chain, host):.8f} "
          f"max|delta|={float(np.abs(chain-host).max()):.4f}")
    check("chain no less accurate than host path",
          cosine(chain, ref) >= cosine(host, ref) - 5e-4,
          f"chain={cosine(chain, ref):.8f} host={cosine(host, ref):.8f}")

    print("\n=== 4. fallback when the device chain fails")
    # injected into the live runtime rather than a second RawRT: two runtimes
    # would compete for the same 6 hw_contexts and the failure under test would
    # be swamped by context thrash
    real = rt._attn_chain_dev

    def boom(*a, **kw):
        raise RuntimeError("injected: hw_context exhausted")

    rt._attn_chain_dev = boom
    t0 = time.perf_counter()
    fb = E.encoder(W, mel, rt.matmul, rt.matmul, prof={},
                   attn_fn=rt.attn_chain)
    fwall = time.perf_counter() - t0
    peak_ctx = max(peak_ctx, len(rt._live))
    check("fallback produced a result", fb.shape == ref.shape)
    check("fallback still passes the gate", cosine(fb, ref) >= COS_FLOOR,
          f"cosine={cosine(fb, ref):.8f} wall={fwall*1e3:.0f} ms")
    check("fallback was recorded", len(rt.fallbacks) == 1,
          "; ".join(rt.fallbacks))
    check("chain disabled after failure", not rt.chain_ok)
    rt._attn_chain_dev = real
    rt.chain_ok = True
    rt.fallbacks.clear()

    print("\n=== 5. hw_context ceiling")
    check("contexts within ceiling", peak_ctx <= rt.max_ctx,
          f"peak={peak_ctx} ceiling={rt.max_ctx}")

    print("\n=== 6. transcription")
    out_path = str(ROOT / "out_test_chain.npy")
    np.save(out_path, chain.astype(np.float32))
    refpy = os.environ.get("NPU_REFENV_PYTHON") or str(
        ROOT / "refenv" / "bin" / "python")
    if not os.path.exists(refpy):
        check("decoder ran", False,
              f"{refpy} not found; set NPU_REFENV_PYTHON to a torch+whisper "
              f"interpreter, or run this inside the container")
        return summary()
    r = subprocess.run([refpy, str(ROOT / "tools" / "decode_with.py"), out_path],
                       capture_output=True, text=True, cwd=ROOT)
    txt = r.stdout
    npu_line = [l for l in txt.splitlines() if l.strip().startswith("'")]
    got = npu_line[-1].strip().strip("'") if npu_line else "<no output>"
    check("decoder ran", r.returncode == 0, r.stderr.strip()[-200:])
    check("transcription matches CPU reference", "\nMATCH" in txt)
    check("transcription is the expected sentence", got == EXPECTED, repr(got))

    return summary()


def summary():
    bad = [n for n, ok, _ in _results if not ok]
    print(f"\n{len(_results)-len(bad)}/{len(_results)} checks passed")
    if bad:
        print("FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
