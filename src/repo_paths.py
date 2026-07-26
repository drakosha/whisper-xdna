#!/usr/bin/env python3
"""Where the checkout keeps its sources, and what it expects from outside.

Everything is anchored on the repository root, so a clone runs wherever it is
put. Two things are deliberately NOT in the tree:

  * `aot/` -- compiled overlays, rebuilt on demand (1-2 min each) and mounted
    from the host by compose so a container restart does not pay for them again.
  * AMD's AIR attention kernels. `kernels/attn_mmul.cc` includes their
    `attn_npu1.cc`, which carries its own licence and is not vendored here. Set
    AIR_DIR to a checkout of the AIR examples to build the fused attention
    path; nothing else in the repo needs it.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kernels"
AOT = Path(os.environ.get("NPU_AOT_DIR") or ROOT / "aot")


def air_dir():
    """The AIR checkout, or a message saying how to get one."""
    air = os.environ.get("AIR_DIR", "")
    if not air or not Path(air).is_dir():
        raise RuntimeError(
            "the fused attention path needs AMD's AIR attention kernels, which "
            "are not vendored here (attn_mmul.cc includes their attn_npu1.cc). "
            "Set AIR_DIR to a checkout that has kernel_fusion_based/ and "
            "dataflow_based/ under it.")
    return Path(air)


def air_includes():
    """Include dirs for a kernel that pulls in AIR sources.

    Deliberately WITHOUT aie_runtime_lib/AIE2: their attn_npu1.cc defines
    getExpBf16 itself and ships its own lut_based_ops.h with the tables inline,
    so having the runtime one on the path is a redefinition:
      error: redefinition of 'getExpBf16'
    """
    from aie.utils import config
    air = air_dir()
    return [config.cxx_header_path(),
            str(Path(config.cxx_header_path()) / "aie_kernels"),
            str(Path(config.cxx_header_path()) / "aie_kernels" / "aie2"),
            str(air / "kernel_fusion_based"), str(air / "dataflow_based")]
