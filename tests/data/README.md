# Test data

## `jfk.wav`

11 s, 16 kHz mono PCM. An excerpt of John F. Kennedy's inaugural address
(20 January 1961) — "ask not what your country can do for you". A work of the
United States federal government, so it carries no copyright in the US
(17 U.S.C. § 105) and is in the public domain.

Taken from the `samples/` directory of
[whisper.cpp](https://github.com/ggml-org/whisper.cpp), which distributes the
same file for the same purpose.

It is the input for everything that needs real audio: `tools/dump_ref.py`
computes the mel and the torch reference from it, and
`tests/test_encoder_golden.py` compares the NPU transcription against the CPU
one on it.

## `golden_chain.npy`

(1500, 384) float32 — the whisper-tiny encoder output for `jfk.wav`, recorded
from a passing run of the device attention chain. Test 2 of the acceptance
suite compares against it bit for bit, so it is tied to one toolchain (XRT
2.21.75, MLIR-AIE v1.3.4, NPU firmware 1.5.5.391). On a different stack it can
differ while the system is healthy; re-record with `--update-golden` after
checking that tests 1 and 6 pass.
