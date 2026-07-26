#!/usr/bin/env python3
"""16 kHz mono float32 out of a PCM WAV, in one place.

The encoder, the reference dumper and the HTTP service all need the same thing;
three copies of it had already drifted apart in their error messages.
"""
import io
import wave

import numpy as np

SR = 16000


def pcm16_mono(raw, expect_sr=SR):
    """Decode WAV bytes to float32 in [-1, 1). Raises ValueError, never asserts.

    Assertions would vanish under `python -O` and let a wrong-format file
    through as silently wrong audio.
    """
    with wave.open(io.BytesIO(raw), "rb") as w:
        nch, sw, sr, n = (w.getnchannels(), w.getsampwidth(),
                          w.getframerate(), w.getnframes())
        frames = w.readframes(n)
    if sw != 2:
        raise ValueError(f"expected 16-bit PCM, got sampwidth={sw}")
    if expect_sr and sr != expect_sr:
        raise ValueError(f"expected {expect_sr} Hz, got {sr}")
    a = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if nch > 1:
        a = a.reshape(-1, nch).mean(axis=1)
    return a, sr, nch


def read_wav(path, expect_sr=SR, verbose=True):
    """Same, from a file."""
    with open(path, "rb") as f:
        a, sr, nch = pcm16_mono(f.read(), expect_sr)
    if verbose:
        print(f"wav: sr={sr} ch={nch} samples={len(a)} dur={len(a)/sr:.2f}s",
              flush=True)
    return a
