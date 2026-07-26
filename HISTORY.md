# How this came to be

The engineering story, kept in one file because the sequence of wrong turns is
more useful than the final code. Every number here was measured on the machine
described in the README; nothing is projected unless it says so.

## Day 1 — getting the hardware to do anything

AMD ships XDNA1 support for Linux, but the documented path runs through a
licensed toolchain. The open path exists and is not obvious: a kernel ≥ 6.14 for
the in-tree `amdxdna` driver, XRT from distro backports, MLIR-AIE/IRON with
peano as the compiler.

The load-bearing package is `python3-xrt`. Without it MLIR-AIE reports "no NPU
runtime device is available", which reads exactly like missing hardware. In
Debian trixie that package name belongs to something else entirely (XRay
Tracer); the real binding only appears in backports. That cost half a day.

Second trap, same shape: without `--ulimit memlock=-1` and `--cap-add IPC_LOCK`
the first buffer `mmap` fails with `EAGAIN`, which also reads as missing
hardware. This one came back a day later when the Dockerfile was written, and
cost time twice.

By the end of day 1 `whisper-tiny`'s encoder ran end to end — all 74 GEMMs on
the array — and decoded to a transcription identical to the CPU reference. So
"there is no open compiler for Phoenix", which is repeated in a few places
online, is false in practice and not only in theory.

## Day 1 — three walls that were not walls

The first performance numbers were bad, and each explanation for why turned out
to be wrong.

**"412 GFLOPS, the hardware is weak."** An artefact of default tile sizes on an
awkward shape. Tuned per shape: 1409–1761 GFLOPS bf16. Tiles must be chosen per
shape, and a shape like `1536x64x1536` wants 32/64/64 while `1536x1536x64` wants
64/64/16 — a 2.5x and 4.6x difference respectively.

**"A fused per-block design is physically impossible — only 3 MB on chip."** The
weights of one large-v3 block are 39 MB, so they cannot be resident. But a fused
design does not need resident weights: what is resident is the tile programs and
the routes, while weights stream from DDR through ObjectFifos during the run.
The real constraint is elsewhere (16 KB of program memory per tile), and it is
not binding.

**"4.16 s is the floor, the wall is measured."** Thirty-three minutes later that
turned out to be measuring our own wrapper, not the silicon: the same 512³ GEMM
costs 1.385 ms through the IRON Python wrapper and 0.267 ms driven straight from
`pyxrt`, and there were 49 calls per block. Dropping to raw XRT was worth ~1.4x
end to end.

The pattern is worth stating because it repeats through the whole project: a
number that measures your own scaffolding is indistinguishable, from the inside,
from a number that measures the hardware.

## Day 2 — the accuracy detour

With the encoder working on `large-v3`, accuracy became the constraint. The
error at 32 layers compounds in a way it does not at `tiny`'s 4 layers, and the
0.999 cosine bar calibrated on `tiny` turned out to be unreachable at depth in
*any* bf16 configuration — even an exact GELU of bf16-rounded input tops out at
0.99493. Correctness moved to character-exact transcription, with cosine kept as
a regression marker.

Then four wrong diagnoses in a row on one kernel:

1. "The stock GELU kernel is fine, softmax is eating the budget." Backwards —
   GELU cost 3x what softmax did, because the row-op rounds the fc1 activation
   to bf16 *before* the non-linearity, and `gelu(round(x)) ≠ round(gelu(x))`.
2. "The residual is the exp lookup table; 6% error there becomes 20% at the
   output." Algebraically impossible — the sensitivity of `x·m/(1+m)` to a
   relative error in `m` is ~1.0, not 3x. The real cause was that the kernel
   approximates *tanh*-GELU, which diverges from exact erf-GELU by 84% at
   x = −5 even with a perfect exp.
3. "So the tanh surrogate is the carrier." Also wrong: measured, it accounts for
   17% of the residual, and it is largest exactly where the output energy is 0.
4. The actual carrier was bf16 **coefficient** precision — and even that turned
   out to be an artefact of the monomial basis, see the README.

Four confident mechanisms, four refutations, one measurement each. The kernel
that came out of it is 2.5x faster than the one before at equal accuracy.

## Day 2 — the comparison was wrong

The number the whole project was judged against — "whisper.cpp encodes in
2.078 s" — was measured at `-ac 512`, while the NPU path was measured at full
context 1500. Different amounts of work. Measured properly, the same binary on
the same clip takes **6880 ms** at `ctx=1500`.

So the "4x slower" that got the track written off twice was really 1.24x, and
half of it existed only in the measurement.

## Day 2 — where the speed actually was

Once the comparison was honest, the remaining gap turned out to be hygiene, not
architecture. In order of size:

- `pyxrt.runlist` had been benchmarked on day 1 and never wired into the
  runtime. Twenty separate submit+wait per layer for the attention heads; of
  1.69 ms per QK launch, 1.25 ms was not compute. **−1400 ms.**
- The row-wise kernels ran on 8 tiles of 16. A direct 16-worker design is
  rejected by the shim ("no ShimNOCTile has sufficient DMA capacity") because
  npu1 has 4 shim NOC tiles with 2 in / 2 out channels; routing one fifo per
  column into a memtile, which then splits across that column's 4 tiles, lifts
  the limit. **−1000 ms.**
- Plain LRU eviction picked the fc2 design — the one holding 32 resident weight
  buffers — once per pass, because it is the last thing a layer touches and ages
  while the next pass's conv frontend opens two contexts. 419 MB of re-upload
  per pass. **−290 ms.**
- `bo.write(a.tobytes())` is two copies on top of the required conversion and
  `bo.read()` is a third. `bo.map()` gives a writable memoryview: 0.064 ms
  median against 0.350 ms. **−208 ms.**
- The MLP was crossing the host boundary four times. Folding bias into the GEMM
  (a column of ones on A, bias as a K row on B) let a fused GELU read fc1's fp32
  accumulator directly and write into fc2's A operand. **−825 ms** across two
  steps, and accuracy went *up* because one rounding disappeared.

Total: 8569 → 3344 ms, with accuracy improving from 0.99322 to 0.99486 against a
format floor of 0.99493.

## Day 2 — four things that did not work, and one that turned out to work

Kept because a negative result with a mechanism is worth as much as a positive
one, and these are the obvious things to try next. int8 was in this list for a
day; the correction is at the end of the section, and it is the more useful
story of the two.

**Overlapping host and device work.** Predicted 25%, delivered 1.4%. Both sides
pull from the same LPDDR5x — the GEMM streams weights while numpy streams
activations — so below roughly a 4:1 host:device ratio, interleaving is *worse*
than serial. There is also ~1.3 ms of fixed cost per overlap point. A corollary
that redirected the rest of the work: **removing host work beats overlapping
it**, by about 3x.

**Splitting a GEMM across CPU and NPU.** With both busy: NPU 1082 → 682 GFLOPS,
CPU 829 → 542 GFLOPS, combined 1224 against 1911 if they were independent — 64%
of the additive estimate. Ceiling 1.13x wall clock for 100% of the CPU, which
defeats the reason for using the NPU at all.

**AMD's own kernel optimisation catalogue** (`skills/aie-kernel-opt` in the
mlir-aie tree). Its levers apply by the letter, and its precedent claims −47%.
On our kernels `AIE_LOOP_UNROLL_FULL` made things **14% slower**. The catalogue's
precedents are convolution kernels with scalar gathers and branches; a straight
vector loop with compile-time constants has nothing to unroll and only gains
register pressure.

**Fused flash-attention.** Built, correct, and still slower — see the README for
the design and the numbers. Two sub-results worth keeping: the reason it was
rejected months earlier ("online rescale needs an fp32 vector multiply, which
AIE2 lacks") was **wrong** — the rescale lives in an fp32 accumulator with bf16
multiplies; and AMD's own driver comments say a 4×4 array can only do one head
at a time, which contradicts the obvious "one column per head" layout.

### int8, and how a negative result got written down wrong

The speed was never in doubt: 1.51x on projections/fc1, 1.36x on fc2, 363 ms per
pass, with per-channel weight scales and per-token activation scales on the 290
projection and MLP GEMMs. What was written down was that the accuracy killed it —
cosine **0.864**, worse than the stock GELU kernel that had already been rejected
as unusable, blamed on whisper's activation outliers (one channel of 1280 sets
the token's scale). Keeping fc2 in fp32 lifted it to 0.968, still 400x the
remaining error budget, so it went into the list above as closed.

Two things were wrong with that.

The smaller one is that the transcript, not cosine, is this project's acceptance
criterion — the README says so about the bf16 path in the same breath — and the
int8 run had never been decoded. The rest of the day had been spent establishing
that a 0.999 cosine bar is a tiny-only artefact, and then a quantisation scheme
was thrown away on a cosine number anyway.

The larger one only surfaced when it *was* decoded. The first decode run produced
garbage in every mode, which looked like a spectacular confirmation, except that
the fp32 control — the identical path with no quantisation at all — produced
garbage too. Nothing that path does can corrupt an unquantised encoder, so the
fault had to be outside the quantisation. It was: the features came from
`large-v3-turbo` and were being decoded with the `large-v3` model. Encoder and
decoder from different checkpoints, and a whisper decoder handed foreign
cross-attention features does not fail, it confabulates. The one number that had
been carried along as a sanity check, `avg_logprob`, was actively misleading: a
degenerate `'Mr. Mr. Mr. …'` repetition loop scored **−0.168** against the
reference's −0.078. A loop is confident. That control is why
`bench/probe_int8_decode.py` runs fp32 first and always.

Decoded correctly, against the torch reference encoder on 27 live Russian panel
recordings (2.5–60 s, about 24 distinct utterances): fp32 control 27/27
character-identical, int8 per-channel **21/27** at cosine 0.784–0.925,
per-tensor 20/27 at 0.769–0.912. Of the six per-channel differences, two are
punctuation only, two are the same politeness form on the same slurred phrase,
one is a tie on an unintelligible word, and one is int8 being *right* where the
fp32 reference is wrong. There is no semantic failure in the set, and the
longest recording carrying real speech (22.6 s, a full sentence) matches at
0.913. The 29 s and 60 s files match too, but both are near-silent — the
reference itself decodes the 60 s one to a single `'и'` — so a match there says
nothing, and it is worth noticing that this degenerate sample is what sets the
top of the cosine range.

So cosine at this depth is not a proxy for the criterion at all — it fell to
0.784 while the text stayed correct. The 363 ms is back on the table, with the
honest caveat that the quality was measured in a numpy simulation of the
quantisation and the speed on the device: a real kernel has bf16 inputs and a
different accumulation order, so it has to be measured again once it exists.
SmoothQuant remains unmeasured, and would still be a separate project with
calibration and host-side scale bookkeeping, i.e. re-adding exactly the host work
that had just been removed.

## What the process looked like

Most of the wrong turns above were caught the same way: a probe that
distinguishes between two explanations, run before writing any kernel code. A
few specifics that generalise:

- **Judge errors where they land, not where they occur.** A GELU kernel with
  lower raw L2 was worse after the fc2 projection, because its error fell in the
  components fc2 amplifies.
- **Probe on captured data, not synthetic.** `N(0, 4)` puts 30% of its mass where
  real fc1 activations put 0.3%, and it hid a kernel defect completely.
- **A systematic shift is a mechanism; noise is symmetric.** The truncating
  rounding mode was found because 94% of deviations shrank the magnitude.
- **Degenerate test inputs can be blind.** `V = ones` passed with a *wrong* V
  layout, because identical columns make a permutation invisible. `K = 0` (which
  forces a uniform distribution, so the answer must be the column mean) caught
  it.
- **Measure the thing you are replacing on the same scale.** The fused attention
  kernel was compared against exact fp64 attention for a week before anyone
  measured where the *existing* path sits on that scale (0.0068) — which is what
  set the actual threshold.

## Status

The encoder is in production use on the machine it was built for. The fused
attention kernel is in the tree, disabled by default, as a documented negative
result. The remaining unexplored levers and their estimated value are listed at
the end of the README.
