#!/usr/bin/env python3
"""Feed a precomputed encoder output (from the NPU) into whisper's CPU decoder
and compare the transcription against the all-CPU reference path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repo_paths import ROOT
import time
import numpy as np
import torch
import whisper

NPU = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "out.npy")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "tiny"
# The mel has to be the one the encoder output was produced from, so the
# reference dump defaults to the same naming tools/dump_ref.py uses.
REF = sys.argv[3] if len(sys.argv) > 3 else str(
    ROOT / ("ref.npz" if MODEL == "tiny" else f"ref_{MODEL}.npz"))

if not Path(REF).exists():
    sys.exit(f"{REF} not found -- run tools/dump_ref.py"
             + ("" if MODEL == "tiny" else f" --model {MODEL}") + " first")
z = np.load(REF)
mel = torch.from_numpy(z["mel"]).unsqueeze(0)          # (1, n_mels, 3000)
model = whisper.load_model(MODEL).eval()
LANG = sys.argv[4] if len(sys.argv) > 4 else "en"
opts = whisper.DecodingOptions(fp16=False, language=LANG, without_timestamps=True,
                               beam_size=None, temperature=0.0)

def decode(feats=None, label=""):
    real_fwd = model.encoder.forward
    if feats is not None:
        model.encoder.forward = lambda x: feats
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            res = whisper.decode(model, mel, opts)
    finally:
        model.encoder.forward = real_fwd   # or a raised error leaves the stub in
    dt = time.perf_counter() - t0
    r = res[0] if isinstance(res, list) else res
    print(f"[{label}] {dt*1e3:.0f} ms  avg_logprob={r.avg_logprob:.4f} "
          f"no_speech={r.no_speech_prob:.4f}")
    print(f"    {r.text!r}")
    return r.text

ref_txt = decode(None, "CPU torch encoder (reference)")

npu = np.load(NPU)
feats = torch.from_numpy(npu).unsqueeze(0).float()
print("npu features:", tuple(feats.shape))
npu_txt = decode(feats, "NPU bf16 encoder")

print("\nMATCH" if ref_txt.strip() == npu_txt.strip() else "\nDIFFER")
