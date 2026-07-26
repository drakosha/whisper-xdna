#!/usr/bin/env python3
"""How many concurrent hw_contexts does NPU1 allow, and what does creating one cost?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repo_paths import AOT as AOT_DIR
import time
import numpy as np
import pyxrt

AOT = str(AOT_DIR)
# Any eight distinct overlays will do -- the question is how many contexts the
# device allows, not what runs in them. Names come from the cache the encoder
# fills, or from the command line.
NAMES = [Path(n).stem for n in sys.argv[1:]] or sorted(
    p.stem for p in AOT_DIR.glob("*.xclbin"))[:8]
if not NAMES:
    sys.exit(f"no overlays in {AOT}: run bench/run_raw_encoder.py once to "
             f"compile some, or pass names as arguments")

dev = pyxrt.device(0)
held = []
for i, nm in enumerate(NAMES):
    try:
        xb = pyxrt.xclbin(f"{AOT}/{nm}.xclbin")
        uuid = dev.register_xclbin(xb)
        t0 = time.perf_counter()
        ctx = pyxrt.hw_context(dev, uuid)
        dt = (time.perf_counter() - t0) * 1e3
        held.append(ctx)
        print(f"  context {i+1:2d} ({nm:6s}) OK   create={dt:.2f} ms")
    except Exception as e:
        print(f"  context {i+1:2d} ({nm:6s}) FAILED: {e}")
        break

print(f"\nmax concurrent hw_contexts held: {len(held)}")

# cost of tearing one down and standing another up
if len(held) >= 1:
    held.pop()
    xb = pyxrt.xclbin(f"{AOT}/ln.xclbin")
    uuid = dev.register_xclbin(xb)
    s = []
    for _ in range(10):
        t0 = time.perf_counter()
        c = pyxrt.hw_context(dev, uuid)
        del c
        s.append((time.perf_counter() - t0) * 1e3)
    print(f"create+destroy one context: min={np.min(s):.2f} ms "
          f"median={np.median(s):.2f} ms")
