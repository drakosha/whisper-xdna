#!/usr/bin/env python3
"""AOT-compile one row-wise elementwise design. Args: op rows cols outdir."""
import sys
from aie.iron.device import from_name
from aie.utils.hostruntime import set_current_device

op, rows, cols, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
valid = int(sys.argv[5]) if len(sys.argv) > 5 else 0
set_current_device(from_name("npu", n_cols=4))
import rowops

design = {"ln": rowops.layer_norm, "sm": rowops.softmax_row,
          "s2": rowops.softmax_row2, "s3": rowops.softmax_row3,
          "s3w": rowops.softmax_row3_win,
          "s4": rowops.softmax_row4, "s4w": rowops.softmax_row4_win, "gl": rowops.gelu_row,
          "g2": rowops.gelu_row2, "g3": rowops.gelu_row3,
          "g4": rowops.gelu_row4, "g4f": rowops.gelu_row4f,
          "g4r": rowops.gelu_row4_rne,
          "s4rw": rowops.softmax_row4r_win}[op]
out_cols = int(sys.argv[6]) if len(sys.argv) > 6 else 0
suffix = "" if rowops.N_CORES == 8 else f"c{rowops.N_CORES}"
if out_cols:
    name = f"{op}_{rows}x{cols}o{out_cols}{suffix}"
    spec = design.specialize(rows=rows, cols=cols, out_cols=out_cols)
elif valid > 0:
    name = f"{op}_{rows}x{cols}v{valid}{suffix}"
    spec = design.specialize(rows=rows, cols=cols, valid=valid)
else:
    name = f"{op}_{rows}x{cols}{suffix}"
    spec = design.specialize(rows=rows, cols=cols)
spec.compile(xclbin_path=f"{out}/{name}.xclbin", inst_path=f"{out}/{name}.bin")
print("compiled", name)
