#!/usr/bin/env python3
"""Dump a whisper encoder's weights + a real mel input + the torch reference.

Runs in refenv (torch + openai-whisper); the NPU/numpy side lives in ironenv
and consumes the .npz this writes.

  python3 tools/dump_ref.py                          # tiny  -> ref.npz
  python3 tools/dump_ref.py --model large-v3-turbo   # 32 layers, ~2.6 GB
  python3 tools/dump_ref.py my.wav --mel-only -o ref_s5.npz

--mel-only writes just the mel and the reference output: the weights do not
change between samples, so a per-clip dump does not need to repeat 2.6 GB of
them. It refuses to overwrite a full dump with a partial one.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import whisper

from audio_io import read_wav
from repo_paths import ROOT

DEFAULT_OUT = {"tiny": "ref.npz"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?",
                    default=str(ROOT / "tests" / "data" / "jfk.wav"))
    ap.add_argument("--model", default="tiny",
                    help="any openai-whisper name: tiny, small, large-v3-turbo")
    ap.add_argument("-o", "--out", default=None,
                    help="default: ref.npz for tiny, ref_<model>.npz otherwise")
    ap.add_argument("--mel-only", action="store_true",
                    help="skip the weights; mel + reference output only")
    args = ap.parse_args()

    out_path = Path(args.out or ROOT / DEFAULT_OUT.get(
        args.model, f"ref_{args.model}.npz"))
    if args.mel_only and out_path.exists() and out_path.stat().st_size > 1 << 30:
        sys.exit(f"{out_path} looks like a full dump ("
                 f"{out_path.stat().st_size / 2**30:.1f} GiB); --mel-only would "
                 f"replace it with a weightless one. Pass -o to write elsewhere.")

    print(f"loading {args.model} ...", flush=True)
    model = whisper.load_model(args.model).eval()
    print("dims:", model.dims, flush=True)

    audio = whisper.pad_or_trim(read_wav(args.wav))
    mel = whisper.log_mel_spectrogram(torch.from_numpy(audio),
                                      n_mels=model.dims.n_mels)
    print("mel:", tuple(mel.shape), mel.dtype, flush=True)

    enc = model.encoder
    with torch.no_grad():
        t0 = time.perf_counter()
        ref = enc(mel.unsqueeze(0))
        cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        ref = enc(mel.unsqueeze(0))
        warm = time.perf_counter() - t0
    print(f"torch encoder: {tuple(ref.shape)} cold={cold*1e3:.0f} ms "
          f"warm={warm*1e3:.0f} ms (threads={torch.get_num_threads()})",
          flush=True)

    out = {"mel": mel.numpy().astype(np.float32),
           "ref_out": ref[0].numpy().astype(np.float32)}
    if not args.mel_only:
        for k, v in enc.state_dict().items():
            out["w." + k] = v.numpy().astype(np.float32)
    del model, enc, ref

    nbytes = sum(v.nbytes for v in out.values())
    free = __import__("shutil").disk_usage(out_path.parent).free
    if free < nbytes * 1.1:
        sys.exit(f"{out_path.parent} has {free/2**30:.1f} GiB free, the dump "
                 f"needs {nbytes/2**30:.1f} GiB")
    print(f"writing {out_path}: {len(out)} arrays, {nbytes/2**30:.2f} GiB",
          flush=True)
    np.savez(out_path, **out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
