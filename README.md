# whisper on the AMD XDNA NPU (Ryzen AI, Phoenix / XDNA1)

Running the OpenAI Whisper encoder on the NPU of an AMD Ryzen 7 PRO 8845HS
under Linux, with open tooling only — no Vitis licence, no Windows, no
account-gated AMD installer.

**It works.** The encoder runs end to end with every GEMM executed on the AIE
array, at whatever size the weights say. On `whisper-tiny` — small enough to be
the acceptance gate — the resulting audio features decode to a transcription
identical, character for character, to the CPU reference.

```
[CPU torch encoder (reference)] 'And so my fellow Americans ask not what your country can do for you ask what you can do for your country.'
[NPU bf16 encoder]              'And so my fellow Americans ask not what your country can do for you ask what you can do for your country.'
MATCH
```

The same code path runs `large-v3-turbo`, which is what the HTTP service in
[serve/](serve/) uses. At full context it is **2.08x faster than the CPU build
shipped on the same machine, using 0.25 of a core instead of 15.3** — it both
beats the clock and frees the cores.

## Install

You need Linux ≥ 6.14 (the in-tree `amdxdna` driver), an XDNA1 NPU at
`/dev/accel/accel0`, docker, and ~12 GB of disk. Nothing else: no Vitis, no
ROCm, no vendor installer.

```bash
# 1. build the image — needs nothing but docker, takes minutes
docker build --build-arg WITH_REFERENCE=1 -t npu-whisper:latest .
#    ...or skip the build and use a published one:
#    export NPU_WHISPER_IMAGE=ghcr.io/drakosha/npu-whisper:latest
#    export NPU_WHISPER_IMAGE=kostryukov/npu-whisper:latest      # Docker Hub

# 2. start the service — the whisper checkpoint downloads itself on first start
RENDER_GID=$(getent group render | cut -d: -f3) docker compose up -d stt

# 3. transcribe something
curl -F file=@tests/data/jfk.wav -F audio_ctx=1500 -F language=en \
     http://localhost:8090/inference
# {"text": "And so my fellow Americans, ask not what your country can do for
#           you, ask what you can do for your country."}
```

The image carries the userspace stack and this checkout, and no model weights:
the whisper checkpoint is fetched on first start into a volume, so the same
image serves whatever `NPU_STT_MODEL` names.

The first start compiles one AIE overlay per GEMM shape (minutes) and caches
them in a volume; later starts are seconds. `docker compose logs -f stt` shows
it. If it fails in a way that looks like missing hardware, the three usual
causes are in [Building the container](#building-the-container).

To work on the encoder rather than serve it, `docker compose up -d npu-whisper`
gives you a shell container with the same stack; see
[Running it](#running-it).

## Hardware and software this was built on

| | |
|---|---|
| CPU / NPU | AMD Ryzen 7 PRO 8845HS (Hawk Point), XDNA1 NPU, 4 columns |
| NPU device | `/dev/accel/accel0`, `[0000:c7:00.1] RyzenAI-npu1` |
| Kernel | 7.0.13 (needs >= 6.14 for the in-tree `amdxdna` driver) |
| Driver | `amdxdna` 0.6.0, NPU firmware 1.5.5.391 |
| Runtime | XRT 2.21.75 (`libxrt-npu2`, `libxrt2`, `python3-xrt` from Debian trixie-backports) |
| Compiler | MLIR-AIE / IRON v1.3.4 + `llvm-aie` (peano) |
| OS | Debian 13, everything inside a `debian:13` container |

The NPU userspace is entirely from distro packages; nothing is built from AMD
source. `python3-xrt` is the load-bearing one — without it MLIR-AIE reports
"no NPU runtime device is available" and it looks like the hardware is missing.

## Does it produce the right answer

Yes. whisper-tiny encoder, all 74 GEMMs on the NPU in bf16 with fp32
accumulation:

| configuration | cosine vs torch fp32 | transcription |
|---|---|---|
| GEMMs on NPU, numpy elementwise | 0.99991 | MATCH |
| + GELU and softmax on NPU (default kernels) | **0.99987** | MATCH |

At `whisper-tiny` depth (4 layers) every configuration transcribes correctly and
sits above 0.999 cosine. The interesting story is at **large-v3** depth, 32
layers, where bf16 rounding compounds and the choice of GELU kernel starts to
matter — see [the GELU kernels](#the-gelu-kernels-the-interesting-part) below.

**On large-v3 the load-bearing criterion is the transcription, not cosine.** A
0.999 cosine bar is calibrated on tiny and is not reachable at 32-layer depth in
*any* bf16 configuration (even a perfect GELU of the bf16-rounded input tops out
at cosine 0.99493). Cosine is kept as a regression marker; correctness is judged
character for character against the CPU `torch` reference on three clips —
English 5.9 s, Russian 5.4 s and Russian 22.6 s — and every kernel here passes
all three.

## Is it fast

The comparison target is the whisper.cpp build actually deployed on this
machine: `large-v3-turbo` q5_0, 16 threads. **Measure it at the same context you
measure yourself at** — this is the single biggest trap in this comparison, and
we fell into it. Same binary, same 22.6 s clip:

| `audio_ctx` | encode | decode | total |
|---|---|---|---|
| 512 | 2165 ms | 3739 ms / **1217 runs** | 6779 ms |
| 1500 | **6880 ms** | 378 ms / 103 runs | 7397 ms |

Quoting "whisper.cpp encodes in 2.1 s" while running your own path at full
context overstates the gap threefold. (The 1217 decode steps at `-ac 512` are
the well-known repetition loop on an utterance longer than the context — visible
right there in the timings.)

Like for like, `large-v3` encoder at `ctx=1500`, 32 layers, real dataflow:

| path | encoder | CPU time | cores busy |
|---|---|---|---|
| whisper.cpp q5_0, 16 threads (deployed) | 6940 ms | ~106 CPU-s | 15.3 of 16 |
| **this project, raw XRT** | **3344 ms** | **~0.84 CPU-s** | **0.25** |

**2.08x faster in wall clock for roughly 125x less CPU work.**

Caveats in both directions: whisper.cpp's number includes the conv frontend and
the decoder's cross-attention KV, which the encoder figure does not; and
whisper.cpp is 5-bit quantised while this path is bf16. So it is a reference
point, not an apples-to-apples race — only the transcription and cosine are
strict.

### Where the 2.56x came from

Nothing architectural. The whole speedup is hygiene, found by decomposing the
profile rather than by having good ideas:

| change | saved | what was actually wrong |
|---|---|---|
| `pyxrt.runlist` for the attention heads | ~1400 ms | it was benchmarked early and never wired into the runtime — 20 separate submit+wait per layer |
| row-wise kernels on 16 tiles, not 8 | ~1000 ms | a direct 16-worker design is refused by the shim ("no ShimNOCTile has sufficient DMA capacity"); routing one fifo per column through a memtile lifts it |
| GELU kernel in a centred basis | −506 ms | see [the GELU kernels](#the-gelu-kernels-the-interesting-part) |
| MLP never leaves the device | −825 ms | fc1 chunks land straight in the GELU overlay's input, GELU writes straight into fc2's A operand |
| eviction that respects resident weights | −290 ms | plain LRU picked the one design holding 32 resident weight buffers — 419 MB of re-upload per pass |
| `bo.map()` instead of `read`/`write` | −208 ms | `bo.write(a.tobytes())` is two copies on top of the conversion; `read()` a third |
| `aie::set_rounding(conv_even)` | 0 ms, **+0.0005 cosine** | the default fp32→bf16 conversion **truncates toward zero**; inherited silently through every kernel |

### What did *not* pay off

Four optimisations were implemented or probed and closed with numbers. They are
listed because a negative result with a mechanism saves the next person a week:

- **Overlapping host work with device work — 1.4%, not the 25% predicted.** Both
  sides pull from the same LPDDR5x: the GEMM streams weights while numpy streams
  activations. Below a 4:1 host:device work ratio, overlapping is *worse* than
  serial, plus ~1.3 ms fixed cost per overlap point.
- **Splitting a GEMM across CPU and NPU — 64% of the additive estimate.** With
  both busy: NPU 1082 → 682 GFLOPS, CPU 829 → 542. Ceiling 1.13x wall for 100%
  of the CPU, which defeats the point.
- **AMD's own `skills/aie-kernel-opt` catalogue — 14% slower on our kernels.**
  Its precedents are convolution kernels with scalar gathers and branches;
  `gelu_row4` is already a straight vector loop with compile-time constants, so
  full unrolling only raises register pressure.
- **Fused flash-attention — built, correct, and still loses.** In the tree,
  disabled by default. See below.

### int8 on the wide GEMMs: closed on cosine, reopened on transcripts

This was a fifth entry in the list above, rejected on a cosine number. The
rejection was wrong, and the way it was wrong is the point.

int8 on the projection and MLP GEMMs (290 of them, attention left in fp32, with
per-token activation scales and per-channel weight scales) is worth 1.51x on
projections/fc1 and 1.36x on fc2 — **363 ms per pass**. It costs cosine: 0.78
to 0.93 against the torch fp32 encoder, well under the 0.99486 the bf16 path
holds. On that number it was closed as unusable.

Re-measured against the acceptance criterion — the transcript — on 27 live
Russian panel recordings, 2.5 s to 60 s (about 24 distinct utterances; a few
short ones are byte-identical repeat tests):

| configuration | transcripts identical to the torch reference | cosine |
|---|---|---|
| fp32 control (same path, no quantisation) | **27 / 27** | ~1.00000 |
| int8, per-channel weights, per-token activations | **21 / 27** | 0.784 – 0.925 |
| int8, per-tensor weights (the coarsest scheme) | **20 / 27** | 0.769 – 0.912 |

Not one of the six per-channel differences is a semantic failure. Two are
punctuation only (`Доброе утро!` → `Доброе утро.`, and one where int8 inserts
the *correct* comma); two are the same politeness form on the same slurred
phrase (`разберись` → `разберитесь`); one is a tie on a word that is
unintelligible in the audio; and in one **int8 is right where the fp32 reference
is wrong** (`Даня расписание` → `Дай мне расписание`). The longest recording
that carries real speech — 22.6 s, a full sentence — matches at cosine 0.913.
The 29 s and 60 s files also match, but they are near-silent and their reference
transcripts are `'Я здесь! Я здесь!'` and a bare `'и'`, so they are evidence of
nothing; the 60 s one is where the 0.925 top of the cosine range comes from.

So cosine is not merely not the criterion here — at this depth it is not a good
proxy for one. Neither is `avg_logprob`: in the broken run described below, a
degenerate `'Mr. Mr. Mr. …'` repetition loop scored −0.168 against a reference
of −0.078, because a loop is *confident*. Only the transcript separates these
cases.

Two things this does not say. The 363 ms is **available, not banked** — the
quality numbers come from a numpy simulation of the quantisation, and a kernel
would have bf16 inputs and a different accumulation order, so device-side
quality has to be re-measured once one exists. And 27 short command-style
utterances in one language are a probe, not a corpus — the same gap the accuracy
entry under [Known limits](#known-limits) admits about the bf16 path.

The reproducer is [bench/probe_int8_decode.py](bench/probe_int8_decode.py). It
runs the fp32 control first and always: the original wrong conclusion came from
decoding `large-v3-turbo` features with the `large-v3` model, which garbles
every mode including the unquantised control — which is exactly how it was
caught.

### Fused attention: a working negative result

`attn_fused.cc` + `attn_fused.py` implement QK→softmax→AV as one kernel: 16
tiles, one head across the array, K and V shipped as a single packed object
(a compute tile has 2 input DMA channels and Q+K+V+O needs three), broadcast
from a memtile, `mmul<4,8,4>` block layouts verified on hardware, tail masking
applied *after* softmax rather than as a `-inf` sentinel before it.

It is correct (rel L2 0.0088 against exact softmax attention, versus 0.0068 for
the three-overlay path it replaces) and it collapses the design set from 6
contexts to 4. It still loses: **1151 ms against 1016 ms**, end to end 3879 vs
3344 ms, cosine 0.99423 vs 0.99486.

The reason is worth knowing before you try it: the accumulators must be fp32.
With their bf16 accumulators, rel L2 is 0.120 — accumulated over 48 blocks, and
the running softmax sum matters more than the output accumulator, because its
error scales the entire row through the final division. fp32 fixes the accuracy
and eats the entire transport saving.

### GEMM throughput

bf16 with fp32 accumulation, 4 columns, best observed:

| shape | GFLOPS |
|---|---|
| 1536x1536x384 | 1761 |
| 1536x5120x1280 | 1514 |
| 1536x1280x1280 | 1409 |
| 1536x64x1536 (skinny, attention) | 386 |

int8 reaches ~1930-2020 GFLOPS, about 1.33x over bf16 — less than the 2x you
might expect.

## The GELU kernels (the interesting part)

GELU on the fc1 output is where bf16 numerics get sharp, and the repository
ships three kernels for it because getting it right took three tries. The lesson
generalises to any activation on this class of hardware.

The real fc1 activations are mean −1.99 with a third of their mass in
`x ∈ [−6, −3)`. Five kernels, measured end to end on large-v3:

| kernel | form | cosine (large-v3) |
|---|---|---|
| `gelu_row.cc` (`gl`) | stock `0.5·x·(1+tanh(u))` | 0.951 — unusable at depth |
| `gelu_row2.cc` (`g2`) | logistic `x·sigmoid(2u)` | 0.99066 |
| `gelu_row3.cc` (`g3`) | erf-GELU, piecewise polynomial, monomial basis | 0.99322 |
| `gelu_row4.cc` (`g4r`) | same, **centred basis** + round-to-nearest-even | 0.99462 |
| **`gelu_row4f.cc` (default)** | **g4r reading fc1's fp32 accumulator directly** | **0.99486** |

- The **stock tanh kernel** dies at depth on catastrophic cancellation: in the
  carrying bin `1 + tanh(u) ≈ 0.04`, and the table-rounded `tanh` loses every
  significant digit there.
- The **logistic form** removes that cancellation, but makes the output directly
  proportional to a table `exp`, and that table (written for softmax) is 5–6%
  off in the argument range GELU needs. It also needs a clamp and can overflow
  `x³`.
- The **polynomial form** — `GELU(x) = relu(x) + q(|x|)`, a degree-4 piecewise
  minimax fit of the *exact* erf-GELU — has no `exp`, no table blind zone, no
  clamp, no cube. It is the default.

Two findings here are worth more than the kernels themselves, because both look
like hardware limits and are not.

**"bf16 coefficients are catastrophic" was an artefact of the basis.** In the
monomial form `a₀ + a₁s + a₂s² + …` on the raw argument, the coefficients span
orders of magnitude and the terms very nearly cancel — the same catastrophic
cancellation as the tanh kernel, just hidden. Post-fc2 error 0.94, and we
"fixed" it by carrying every coefficient as a hi/lo bf16 pair. Map each piece to
`u ∈ [−1,1]` instead: state stays O(1), coefficients O(0.1), and **plain bf16
coefficients are fine** (0.00681 vs 0.00663 for exact ones). The hi/lo pairs
become unnecessary, and three degree-4 pieces hold the accuracy that needed five
— 12 Horner steps instead of 20. The kernel got **2.5x faster at equal
accuracy** by changing coordinates, not by optimising code.

**The default rounding mode truncates toward zero.** `aie::set_rounding(conv_even)`
is one line and it moved end-to-end cosine 0.99411 → 0.99462 at identical wall
time, closing 62% of the remaining gap to the bf16 floor. This had been
inherited silently through every kernel in the series. It surfaced only because
a discrepancy was *decomposed instead of dismissed as noise*: 94% of the
deviations shrank the magnitude, and noise is symmetric while a systematic shift
is not. Note the same trick **breaks** softmax (cosine 0.879) — the stock
`getExpBf16` table is calibrated for the default mode.

Two methodological notes that cost us time: errors must be judged **after the
fc2 projection**, not on the raw GELU output — a kernel with lower raw L2 can be
worse once projected; and a synthetic input distribution hid the stock kernel's
defect entirely, because real fc1 activations put 46% of their mass where the
synthetic ones put 0.3%. Probe on captured activations, not on `N(0, 4)`.

`GELU_OP=g4`/`g3`/`g2`/`gl` select the older kernels for A/B.

## Attention chaining on device

`QK → softmax → AV` runs without a round trip to host memory: the QK GEMM writes
bf16 straight into a sub-buffer that the softmax normalises in place and the AV
GEMM reads back, no `sync` between stages. Two facts make it safe — a buffer from
one overlay is accepted as input (and output) by another even across differing
`hw_context`s, and QK's bf16 output is bit-identical to fp32 here (K=64 at
tile 64 is a single accumulation step). On real large-v3 at full context (1500)
the chained path is **8.7 s vs 15.1 s for the host-routed path, 1.73x**, because
the activation transfers dominate at that size. Any failure — xclbin load,
context exhaustion, timeout — falls back transparently to the host path.

Note this is *chaining*, not fusion: the 1500×1500 score matrix still exists,
still 94 MB per layer against 3 MB of on-chip memory, so it still spills to DDR
— roughly 12 GB of traffic per pass. Removing that was the point of the fused
kernel above, which is why it is disappointing that fusing did not win.

Switching between overlays costs **2.81 ms per pair of switches**, measured by
running the same runlist alone (6.708 ms) and with one foreign launch interleaved
(10.516 ms, against 7.709 ms for the two run separately). With three overlays per
layer this is ~107 ms per pass that only fusion can remove.

## What is in here

Everything is driven from Python. There are two backends: the IRON one (simple,
slow) and the raw-XRT one (bypasses the IRON wrapper, ~1.4x faster end to end).

The encoder itself is size-agnostic: `n_layer`, `d_model` and `n_head` are read
off the weights, and overlays are compiled for whatever shapes come out. Two
models are actually exercised — **`whisper-tiny` is the fast gate** (4 layers,
the whole encoder in seconds, a bit-identical golden and a transcription to
compare), and **`large-v3-turbo` is what the service runs** and what every
timing in this README was measured on.

### `src/` — the encoder and its runtime

| file | what |
|---|---|
| `npu_whisper_encoder.py` | the whisper encoder; conv frontend via im2col, attention, MLP. Matmul backend is pluggable |
| `rawxrt.py` | raw-XRT backend: LRU-managed hw_contexts, device-resident weight buffers, on-demand AOT compile, row-ops, the attention chain |
| `rowops.py`, `compile_rowop.py` | row-wise elementwise designs and their AOT compile driver |
| `blocklayout.py`, `mmul_layout.py` | the host-side block layouts AMD's mmul expects, and the smallest check that ours match |
| `attn_fused.py` | QK→softmax→PV as one design (off by default; needs AMD's AIR kernels, see below) |
| `repo_paths.py` | where the checkout keeps kernels and the overlay cache |

### `kernels/` — C++ for AIE2

| file | what |
|---|---|
| `layer_norm_gb.cc` | layernorm for aie2. The stock one is aie2p-only and porting it hits `undefined symbol: sqrtf`, so this uses a Newton-Raphson reciprocal square root |
| `softmax_row.cc` … `softmax_row4_rne.cc` | row-wise softmax with max subtraction; v2 splits the reciprocal into two bf16 halves, v3 the exponentials as well, v4 drops the second `exp`, `_rne` rounds to nearest even |
| `gelu_row.cc` … `gelu_row4f.cc` | the five GELU kernels above; **`gelu_row4f.cc` is the default** |
| `gelu_poly_coeffs.h`, `gelu_poly4_coeffs.h` | the fitted polynomial coefficients the `g3` and `g4` kernels include |
| `attn_mmul.cc`, `attn_fused.cc`, `attn_g_b_f32.inc` | the fused-attention cores |

### `tools/` — reference data and code generation

| file | what |
|---|---|
| `dump_ref.py` | pulls a checkpoint's encoder weights, computes the mel, produces the torch fp32 reference. `tiny` by default (`ref.npz`, 36 MB); `--model large-v3-turbo` is the same for 32 layers (~2.6 GB) |
| `decode_with.py` | feeds encoder output into Whisper's CPU decoder and diffs the transcription |
| `gen_gelu_coeffs.py`, `gen_gelu4_coeffs.py` | fit the `g3` / `g4` polynomials and emit the headers the kernels include |

### `bench/` — timings and the probes behind the claims

`run_raw_encoder.py` is `whisper-tiny` end to end on the raw-XRT backend;
`run_raw_block.py` is one large-v3 encoder block with real dataflow (the honest
per-block number); `layernorm_aie2.py` measures the standalone layernorm design.
The runtime-characteristic probes are the measured backing for the claims above:
`probe_ctx.py` (hw_context ceiling), `probe_switch.py` / `probe_switch_n.py`
(design-switch cost), `probe_floor.py` (per-shape GEMM rate),
`probe_xrt_chain.py` (per-call latency, chaining), `probe_bo_share.py`
(cross-design buffer reuse). `probe_int8_decode.py` is the odd one out: no NPU,
it simulates int8 quantisation in numpy over a corpus of wavs and judges it on
the decoded transcript — see [int8 above](#int8-on-the-wide-gemms-closed-on-cosine-reopened-on-transcripts).

### `tests/` — the acceptance gate

`test_encoder_golden.py` runs on `whisper-tiny`: bit-identical golden regression
against `data/golden_chain.npy`, the attention chain vs the host path, fallback
injection, the context ceiling, and the transcription of `data/jfk.wav`.
`test_rowops.py` checks the row-wise kernels on their own, `test_chain_acc.py`
the accuracy cost of chaining QK→softmax→AV.

### Not in here

The fused attention path (`NPU_FUSED_ATTN=1`) includes AMD's AIR attention
kernels, which are not vendored: point `AIR_DIR` at a checkout of them. The
GEMM-with-GELU-fused design that `--fused-gelu` wants is not in here either
(`NPU_FUSED_PY`); it lost to the row-wise GELU on accuracy anyway.

## Building the container

The `Dockerfile` builds the whole userspace stack from distro packages and
upstream releases — XRT from trixie-backports, MLIR-AIE/IRON v1.3.4 with peano,
and this checkout. It does not and cannot contain the driver: `amdxdna` has been
in-tree since Linux 6.14 and comes from the host kernel.

```bash
docker build -t npu-whisper .
docker run --rm -it \
  --device /dev/accel/accel0 --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --cap-add IPC_LOCK --ulimit memlock=-1 \
  npu-whisper
```

Three things are load-bearing, and each fails in a way that looks like missing
hardware:

- `/dev/accel/accel0` is owned by group `render`; without `--group-add` the
  device node is there but unopenable.
- XRT locks its device buffers, so `--cap-add IPC_LOCK --ulimit memlock=-1` are
  required — without them the first buffer `mmap` fails with `EAGAIN` and the
  NPU looks absent.
- No privileged mode is needed.

`compose.yml` does the same with named volumes for the compiled-kernel cache
and the whisper checkpoint, so both survive the container being recreated, and
carries a second service, `stt`, which starts the HTTP endpoint instead of a
shell. The reference
environment (torch + openai-whisper + ffmpeg, ~1.9 GB) is off by default; the
tools under `tools/` and everything in `serve/` need it, so add it with
`--build-arg WITH_REFERENCE=1`.

Two sanity checks, in order — the first needs no NPU, the second does:

```bash
docker run --rm npu-whisper python3 -c "import aie.iron, pyxrt; print('ok')"
docker run --rm --device /dev/accel/accel0 --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  --cap-add IPC_LOCK --ulimit memlock=-1 npu-whisper \
  python3 -c "import pyxrt; print(pyxrt.device(0).get_info(pyxrt.xrt_info_device.name))"
# expected: RyzenAI-npu1
```

## Serving it

`serve/` wraps the encoder in an HTTP endpoint that speaks the same multipart
contract as `whisper.cpp`'s `/inference`, so an existing client can be pointed at
it unchanged — that is what [Install](#install) starts. `NPU_STT_MODEL` picks
the checkpoint, `large-v3-turbo` by default; both halves come out of that one
file, so the encoder on the NPU and the decoder on the CPU can never drift
apart. Measured on this machine at ctx=1500, one 11 s English clip:
`tiny` 484 ms encoder, `small` 1507 ms, `large-v3-turbo` 3346 ms, `large-v3`
3990 ms, each transcribing correctly.

See [serve/README.md](serve/README.md) — in particular the two failure modes
that cost a day: changing `audio_ctx` between requests kills the NPU path unless
the old geometry is unloaded, and the obvious fix for *that* returns **silently
wrong text** with entirely normal timings.

## Running it

Two Python environments: `ironenv` (MLIR-AIE, numpy only) and a separate
`refenv` (torch + openai-whisper) used only to produce the reference and decode.

```bash
# 1. reference: whisper-tiny weights, mel, torch fp32 encoder output -> ref.npz
./refenv/bin/python tools/dump_ref.py tests/data/jfk.wav

# 2. the encoder, end to end on the NPU
python3 bench/run_raw_encoder.py --iters 3 --npu-gelu --save out.npy

# 3. does it still transcribe correctly
./refenv/bin/python tools/decode_with.py out.npy

# 4. one large-v3 block with real dataflow (the honest per-block number)
python3 bench/run_raw_block.py --iters 5 --npu-gelu

# 5. the acceptance gate
python3 tests/test_encoder_golden.py
```

`PEANO_INSTALL_DIR` must point at the `llvm-aie` wheel for anything that
compiles a design. If MLIR-AIE lives somewhere other than `/opt/mlir-aie`, set
`MLIR_AIE_DIR`.

## Gotchas found the hard way

- **`C_t.numpy()` returns a view onto the XRT buffer.** If the tensor is a local
  it is freed on return and the view dangles — the next numpy operation
  segfaults. Copy out before returning.
- **NPU1 allows exactly 6 concurrent `hw_context`s.** The 7th fails with
  `DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22)`. Since every distinct
  GEMM shape is its own overlay, this is a real planning constraint; exceeding it
  costs ~2.5 ms per context create in thrash.
- **`pyxrt.runlist` works on NPU1** (~24% better per run) even though the
  MLIR-AIE examples mark it NPU2-only.
- **Chaining is free**: feeding one run's output buffer straight into the next as
  input costs nothing extra — no sync, no read, no copy.
- **The IRON Python wrapper costs ~1.1 ms per call.** The same 512³ GEMM is
  1.385 ms through IRON and 0.267 ms through pyxrt directly.
- **The bf16 exp table (`getExpBf16`) aliases past its valid range.** It was
  written for softmax and returns garbage below argument ≈ −142, which shows up
  as periodic `inf` bands, not a clean saturation.
- **A stale `.prj` build directory silently defeats a kernel edit** — IRON
  relinks its object files from it; delete the `.prj` after changing a kernel.

## Known limits

- Only the encoder is on the NPU. The decoder stays on the CPU, and not only
  because it is small for `large-v3-turbo` (~63 ms of a 2234 ms recognition) —
  it is the wrong shape for this device. Decoding is autoregressive, one token
  per step, so every GEMM degenerates to a matrix-vector product: ~815 M weights
  read per token, 1.6 GB in bf16, against ~1.7 GFLOP of arithmetic. That is
  ~1.1 FLOP/byte against a machine balance point near 15 — deeply memory-bound,
  on an NPU that shares the LPDDR5x with the CPU (the same contention that held
  host/device overlap to 1.4% instead of the predicted 25%, above). Three
  further multipliers, all against: the deployed whisper.cpp reads q5_0
  (~640 MB/token) where an NPU path would read bf16, a threefold loss on a
  workload where time *is* bytes; per-call cost is ~1.1–1.3 ms with at least one
  call per layer per token, so 32 × 1.2 ms ≈ 38 ms per token against a 63 ms
  decoder for the whole recording; and the KV cache grows every token while
  MLIR-AIE compiles static shapes. Batching or speculative decoding would raise
  the arithmetic intensity enough to change this — neither applies at batch 1,
  which is what a single-user endpoint is.
- Accuracy on large-v3 is validated by transcription on three clips, not by a
  cosine bar (which is a tiny-only artefact); a large corpus with CER/WER would
  be a stronger claim than three clips.
- One fused overlay per transformer block was not built, but the reasons are now
  measured rather than assumed: not on-chip weight storage (weights stream from
  DDR through ObjectFifos), not program memory (a block-sized kernel lands near
  10 KB against a 16 KB per-tile limit) — what remains is the dataflow plumbing
  between stages and the 6-context ceiling.
- Shapes are padded to tile granularity, so sequence length 1500 is computed as
  1536; the padding is stripped before softmax.

## What else exists, and why it does not cover this case

Whisper on Ryzen AI is not new. The opening claim is narrower than that: *Linux,
XDNA1, open tooling only*. Everything below was checked in July 2026.

- **AMD's whisper.cpp fork** — [amd/whisper.cpp](https://github.com/amd/whisper.cpp),
  documented at
  [ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/latest/whisper_cpp.html).
  Full encoder offload, and the closest thing to this project in intent. Its own
  documentation: "NPU acceleration is currently supported on Windows only, with
  Linux support planned." It targets the Ryzen AI 300 Series, i.e. XDNA2. This
  is Linux on XDNA1.
- **AMD's RyzenAI-SW ASR demo** —
  [RyzenAI-SW/Demos/ASR/Whisper](https://github.com/amd/RyzenAI-SW/tree/main/Demos/ASR/Whisper).
  base/small/medium plus `large-v3-turbo`, in **BFP16** ("near-FP32 accuracy at
  INT8-like performance"). It needs the Ryzen AI SDK, a `ryzen-ai-<version>`
  conda environment and ONNX Runtime with the VitisAI execution provider — the
  account-gated stack the first paragraph of this README rules out. If you are
  willing to install it, this is the supported path and you should use it.
- **Third-party int8 checkpoints**, e.g.
  [magicunicorn/whisper-large-v3-amd-npu-int8](https://huggingface.co/magicunicorn/whisper-large-v3-amd-npu-int8)
  and its siblings. These claim WER 1.0% on LibriSpeech test-clean, RTF 0.0045,
  "220x faster than CPU", with both encoder *and* decoder on the NPU of a Ryzen
  7040/8040. Two problems. The repositories contain no weights — `.gitattributes`,
  `README.md` and `config.json`, 11.8 kB in total. And the speed claim does not
  survive arithmetic: the large-v3 encoder is ~2.26 TFLOP per 30 s window
  (32 layers × ~70.5 GFLOP — QKVO `4×2·1500·1280·1280`, MLP
  `2×2·1500·1280·5120`, QKᵀ and AV `2×2·1500·1500·1280`), so an hour of audio is
  ~271 TFLOP, which at the 16 TOPS int8 peak of this class of part is ~16.9 s
  **at 100% of peak, encoder only** — against a claimed 16.2 s that also
  includes a decoder. The WER is quoted without an fp32 baseline on the same
  harness.

## Not in the repository

These are generated or downloaded, and large:

| | why |
|---|---|
| compiled overlays (`*.xclbin`, `*.bin`) | rebuilt on demand (~1-2 min each); kept in the `aot` volume |
| `ref.npz` | 36 MB, produced by `tools/dump_ref.py` |
| `ref_large-v3-turbo.npz` | ~2.6 GB, produced by `tools/dump_ref.py --model large-v3-turbo`; only for offline large-v3 accuracy checks |
| `out*.npy` | encoder outputs, produced by a run |
| `refenv/` | ~1.9 GB virtualenv (torch + openai-whisper) |
| whisper checkpoints | downloaded on first use (`large-v3-turbo` is ~1.6 GB); kept in the `models` volume |
| AMD's AIR attention kernels | needed only by the fused attention path; point `AIR_DIR` at a checkout |

## Dependencies and licences

Everything original here is **MIT** (see `LICENSE`). Files derived from
upstream projects keep the licence they came with, and `NOTICE` lists them one
by one; the short version:

| what | licence |
|---|---|
| this repository | MIT |
| row-wise AIE2 kernels (`gelu_row*`, `softmax_row*`, `layer_norm_gb.cc`) | Apache-2.0 WITH LLVM-exception, from [MLIR-AIE](https://github.com/Xilinx/mlir-aie)'s `aie_kernels/` |
| attention kernels (`attn_mmul.cc`, `attn_fused.cc`, `attn_g_b_f32.inc`) | MIT, © 2025 AMD, from [MLIR-AIE's AIR examples](https://github.com/Xilinx/mlir-air) |
| `tests/data/jfk.wav` | see [tests/data/README.md](tests/data/README.md) |

Fetched at build time and not redistributed here:

| component | licence |
|---|---|
| [MLIR-AIE / IRON](https://github.com/Xilinx/mlir-aie) | Apache-2.0 WITH LLVM-exception |
| [llvm-aie (peano)](https://github.com/Xilinx/llvm-aie) | Apache-2.0 WITH LLVM-exception |
| [XRT](https://github.com/Xilinx/XRT) | Apache-2.0 |
| `amdxdna` kernel driver | GPL-2.0 (in-tree since Linux 6.14) |
| [openai-whisper](https://github.com/openai/whisper) | MIT (reference + decoder only) |
| PyTorch | BSD-3-Clause (reference only) |

One deliberate exclusion: MLIR-AIE also ships `programming_examples/ml/magika`,
which contains a Phoenix layernorm under `LicenseRef-AMD-Proprietary`. Nothing
here derives from it — `layer_norm_gb.cc` is written against the Apache-licensed
aie2p kernel instead.
