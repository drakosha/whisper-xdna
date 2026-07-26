#!/usr/bin/env python3
"""int8 quantisation judged on the transcript, not on cosine.

For every .wav in NPU_SAMPLES: mel -> torch encoder (the reference) -> the same
encoder with a quantised numpy matmul, once per mode -> decode both with the CPU
decoder -> compare the two transcripts character for character.

MATCH means the transcript decoded from the quantised features equals the one
decoded from the torch reference encoder for that same clip. It does not mean
equality with the production transcript; that is printed alongside for human
context only, since production runs a different build with its own audio_ctx and
is itself sometimes wrong.

The fp32 mode is a control -- the identical path with no quantisation -- and it
runs first and always. It is what catches an environment fault, of which the
expensive one is decoding features of one checkpoint with another checkpoint's
decoder: the decoder does not fail on foreign cross-attention features, it
confabulates, and it does so with a perfectly healthy avg_logprob.

  NPU_REF_LV3   .npz from tools/dump_ref.py --model large-v3-turbo
  NPU_SAMPLES   directory of 16 kHz mono .wav, each with a sibling .txt
  NPU_MODES     comma-separated: fp32, perchan, pertensor  (default fp32,perchan)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repo_paths import ROOT
import os
import time
import wave
import numpy as np
import torch
import whisper

import npu_whisper_encoder as E

MODEL = "large-v3-turbo"       # must be the checkpoint the reference was dumped
                               # from, or every mode decodes to fluent nonsense
REF = Path(os.environ.get("NPU_REF_LV3", ROOT / f"ref_{MODEL}.npz"))
SAMPLES = Path(os.environ.get("NPU_SAMPLES", ROOT / "samples"))
MODES = os.environ.get("NPU_MODES", "fp32,perchan").split(",")
LANG = os.environ.get("NPU_LANG", "ru")
STAT = {"n": 0}


def q_matmul_factory(mode):
    """A matmul that quantises A and B to int8 before multiplying.

    Per-token scales on the activations, per-channel (`perchan`) or single
    (`pertensor`) scales on the weights. `fp32` quantises nothing and is the
    control.
    """
    def q_matmul(A, B, bias=None, **kw):
        A = np.ascontiguousarray(A, np.float32)
        B = np.ascontiguousarray(B, np.float32)
        if mode == "fp32":
            r = A @ B
            return r if bias is None else r + bias
        sa = np.abs(A).max(axis=1, keepdims=True) / 127.0
        sa[sa == 0] = 1.0
        if mode == "pertensor":
            sw = np.full((1, B.shape[1]), np.abs(B).max() / 127.0, np.float32)
        else:
            sw = np.abs(B).max(axis=0, keepdims=True) / 127.0
        sw[sw == 0] = 1.0
        qa = np.rint(A / sa).astype(np.int8)
        qb = np.rint(B / sw).astype(np.int8)
        r = (qa.astype(np.float64) @ qb.astype(np.float64)).astype(np.float32)
        r = r * sa * sw
        STAT["n"] += 1
        return r if bias is None else r + bias
    return q_matmul


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        nch, sr, n = w.getnchannels(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    assert sr == 16000, f"{path}: sr={sr}, want 16000"
    return a, n / sr


if not REF.exists():
    sys.exit(f"{REF} not found -- run tools/dump_ref.py --model {MODEL}, "
             "or point NPU_REF_LV3 at an existing dump")
if not SAMPLES.is_dir():
    sys.exit(f"{SAMPLES} not found -- point NPU_SAMPLES at a directory of wavs")

print(f"loading weights from {REF} ...", flush=True)
z = np.load(REF)
W = {k: z[k].astype(np.float32) for k in z.files}
W.pop("mel"), W.pop("ref_out")
n_layer = 1 + max(int(k.split(".")[2]) for k in W if k.startswith("w.blocks."))
d_model = W["w.ln_post.weight"].shape[0]
n_head = d_model // 64
for k in [k for k in W if k.endswith(".weight") and W[k].ndim == 2
          and not k.endswith("_ln.weight")]:
    W[k[:-len("weight")] + "weight_T"] = np.ascontiguousarray(W[k].T)
E.prepare_mlp_chunks(W, n_layer, d_model, 0)
print(f"weights ready: {n_layer} layers, {n_head} heads", flush=True)

# One load for the whole run: the model is 1.6 GB and the decoder is the same
# for every sample and every mode.
model = whisper.load_model(MODEL).eval()
# Greedy, no temperature fallback, language fixed rather than autodetected --
# exact string comparison is meaningless against a sampling decoder.
opts = whisper.DecodingOptions(fp16=False, language=LANG, without_timestamps=True,
                               beam_size=None, temperature=0.0)

wavs = sorted(SAMPLES.glob("*.wav"))
print(f"model ready, {len(wavs)} samples, modes: {','.join(MODES)}\n", flush=True)

tally = {m: [0, 0] for m in MODES}          # mode -> [match, total]
for path in wavs:
    prod_txt = path.with_suffix(".txt")
    prod = prod_txt.read_text().strip() if prod_txt.exists() else ""
    try:
        audio, dur = read_wav(path)
    except Exception as e:
        print(f"### {path.stem}: unreadable ({e})\n", flush=True)
        continue

    mel = whisper.log_mel_spectrogram(torch.from_numpy(whisper.pad_or_trim(audio)),
                                      n_mels=128)
    print(f"### {path.stem}  dur={dur:.1f}s")
    print(f"    prod : {prod[:110]!r}")

    real_fwd = model.encoder.forward
    with torch.no_grad():
        ref = model.encoder(mel.unsqueeze(0).float())[0].numpy()
        r = whisper.decode(model, mel.unsqueeze(0), opts)
    ref_txt = (r[0] if isinstance(r, list) else r).text.strip()
    print(f"    torch: {ref_txt[:110]!r}")

    for mode in MODES:
        STAT["n"] = 0
        t0 = time.perf_counter()
        out = E.encoder(W, mel.numpy().astype(np.float32), q_matmul_factory(mode),
                        E.np_matmul, n_layer=n_layer, n_head=n_head,
                        fold_bias=False)
        dt = time.perf_counter() - t0
        cos = float((out * ref).sum() /
                    (np.linalg.norm(out) * np.linalg.norm(ref)))
        feats = torch.from_numpy(out).unsqueeze(0).float()
        model.encoder.forward = lambda x, _f=feats: _f
        try:
            with torch.no_grad():
                rr = whisper.decode(model, mel.unsqueeze(0), opts)
        finally:
            model.encoder.forward = real_fwd   # or a raise leaves the stub in
        rr = rr[0] if isinstance(rr, list) else rr
        txt = rr.text.strip()
        ok = txt == ref_txt
        tally[mode][0] += ok
        tally[mode][1] += 1
        print(f"    {mode:9s} cos={cos:.5f} gemms={STAT['n']:3d} {dt:4.0f}s "
              f"logp={rr.avg_logprob:+.4f} {'MATCH ' if ok else 'DIFFER'} "
              f"{txt[:90]!r}", flush=True)
    print(flush=True)

for mode in MODES:
    m, n = tally[mode]
    note = "  <- control, must be all-MATCH" if mode == "fp32" else ""
    print(f"{mode:9s} {m:3d} / {n:<3d} MATCH{note}")
