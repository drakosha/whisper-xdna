#!/usr/bin/env python3
"""STT over the XDNA1 NPU encoder, speaking whisper.cpp's HTTP contract.

Pipeline: multipart audio -> ffmpeg/wav -> log-mel -> NPU encoder -> torch
CPU decoder (large-v3-turbo) -> {"text": ...}.

Runs in refenv, which is where torch and openai-whisper live. rawxrt needs only
numpy, ml_dtypes and pyxrt at run time -- it shells out to ironenv for AOT
compiles -- so both halves fit in ONE process. That is measured, not assumed:
under this interpreter tests/test_encoder_golden.py is 14/14 bit-identical and
the full-context cosine is 0.99486399, the same eight digits as ironenv.

Start it with serve/run.sh, which sets the two paths it needs and picks refenv.

Encoder weights come from the whisper checkpoint itself: on the first start
openai-whisper downloads large-v3-turbo (~1.6 GB) into its cache, the encoder
half is lifted out as fp32 numpy and uploaded to the device, and the torch
encoder is dropped before the decoder ever runs. NPU_STT_WEIGHTS overrides that
with an .npz from tools/dump_ref.py --model, for an air-gapped box.

Endpoints
  POST /inference                  whisper.cpp form: file, response_format,
                                   audio_ctx, model, language
  POST /v1/audio/transcriptions    same handler, OpenAI-ish path
  GET  /health                     json status, no device work

With STT_CPU_URL set, a failed or busy NPU is proxied to whisper.cpp instead of
being reported to the client -- the inner half of a two-layer fallback. The
outer half stays in the client, which goes to the whisper.cpp endpoint on its
own when this service is not answering at all; a fallback that lives only in
here dies with the process it lives in.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from audio_io import pcm16_mono          # noqa: E402  (needs the path above)

PORT = int(os.environ.get("NPU_STT_PORT", "8090"))
HOST = os.environ.get("NPU_STT_HOST", "0.0.0.0")
# Empty = take the encoder weights from the whisper checkpoint, downloading it
# on first start. A path here loads an .npz from tools/dump_ref.py instead.
WEIGHTS = os.environ.get("NPU_STT_WEIGHTS", "").strip()
MODEL = os.environ.get("NPU_STT_MODEL", "large-v3-turbo")
LANG = os.environ.get("NPU_STT_LANG", "ru")
# Every distinct context is its own set of compiled overlays, and a client that
# computes audio_ctx from speech length sends a continuum. Round UP onto
# a fixed ladder: bounded overlay set, warm at start, never less context than
# the client asked for.
# Sorted, because ctx_for() walks it to round a request UP; an unsorted list
# would quietly hand back less context than asked for, which makes the decoder
# loop on the tail.
LADDER = sorted(int(x) for x in
                os.environ.get("NPU_STT_LADDER",
                               "512,768,1024,1280,1500").split(",") if x.strip())
if not LADDER:
    raise SystemExit("NPU_STT_LADDER is empty: give at least one context")
# audio_ctx absent (a client that does not know the length sends none): take the
# full context. whisper.cpp's own default is -ac 512, but a 512 cap on
# unknown-length speech is what makes the decoder loop on the tail.
DEFAULT_CTX = int(os.environ.get("NPU_STT_DEFAULT_CTX", "1500"))
# Measured: the decoder's wall clock is flat across 4/8/16 threads
# -- 103 sequential steps, latency-bound, not throughput-bound -- but the CPU it
# burns is not, and 16 costs the least (6.2 CPU-s on the 22.6 s sample against
# 9.0 at four and 15.9 at eight).
TORCH_THREADS = int(os.environ.get("NPU_STT_TORCH_THREADS", "16"))
# A queued request must not outlive the client, which typically gives it 30 s.
QUEUE_WAIT_S = float(os.environ.get("NPU_STT_QUEUE_WAIT", "20"))
MAX_BODY = int(os.environ.get("NPU_STT_MAX_BODY", str(64 << 20)))
# A request holds the device for the whole recording, one 30 s window at a
# time, so a long upload blocks every other caller for the duration: measured,
# 5 minutes of speech is 10 windows and 39 s of work. 20 windows caps that at
# ~80 s; 0 lifts the cap entirely.
MAX_WINDOWS = int(os.environ.get("NPU_STT_MAX_WINDOWS", "20"))
# Transcripts are user speech. Off by default; set to 1 when debugging.
LOG_TEXT = os.environ.get("NPU_STT_LOG_TEXT", "0") != "0"
PREWARM = os.environ.get("NPU_STT_PREWARM", "1") != "0"
FFMPEG = os.environ.get("NPU_STT_FFMPEG", "ffmpeg")

SR = 16000
# Overwritten from the checkpoint's own dims at load: large-v3 and turbo use
# 128 mel bins, everything smaller uses 80, and feeding the wrong count is a
# shape error at the first convolution.
N_MELS = 128
N_SAMPLES = 30 * SR                       # one whisper window

# A second audio_ctx in the same process blows the hw_context ceiling
# (DRM_IOCTL_AMDXDNA_CREATE_HWCTX err=-22) unless the previous geometry is
# released first, and rawxrt then disables the device attention chain and MLP
# for good -- 3.3 s becomes 13 s. Measured: a lower ceiling (max_ctx=5) does not
# help and dropping only the attention chain does not help; releasing every
# design of the old geometry does, with zero
# fallbacks and the same steady-state timings as a single-context process.
RECOVER = os.environ.get("NPU_STT_RECOVER", "1") != "0"

# ---------------------------------------------------------------- CPU fallback
# Inner layer of a two-layer scheme. This one hides a failed or busy NPU from
# the client by proxying the ORIGINAL request to whisper.cpp. The outer layer
# stays where it is -- in the client, which goes to whisper.cpp itself when this
# whole service is down -- because a fallback that lives only here dies with the
# process it lives in.
#
# Empty = do not fall back, which is the default: a fresh checkout has no CPU
# endpoint to proxy to, and deploying this must not silently reroute traffic.
CPU_URL = os.environ.get("STT_CPU_URL", "").strip()
CPU_TIMEOUT = float(os.environ.get("STT_CPU_TIMEOUT", "25"))
# How long a request waits for a busy device before giving up on it. Zero is
# measured, not assumed: waiting pays only while
# wait + npu_latency < cpu_latency, which is a 205 ms budget on short speech,
# and the odds of the in-flight request finishing inside it are ~10%.
BUSY_WAIT_MS = float(os.environ.get("STT_BUSY_WAIT_MS", "0"))
# Test hook: make every NPU attempt fail, so the failure path can be exercised
# without breaking the device. Off unless explicitly set.
FAIL_INJECT = os.environ.get("NPU_STT_FAIL_INJECT", "0") != "0"

_lock = threading.Semaphore(1)
_counters = threading.Lock()      # _state is written from handler threads
_state = {"ready": False, "requests": 0, "errors": 0, "started": time.time(),
          "ctx": None, "teardowns": 0, "degraded": False,
          "served_npu": 0, "fb_busy": 0, "fb_error": 0, "fb_decode": 0,
          "fb_failed": 0, "degradations": 0}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bump(*names, **kv):
    """Counters, atomically -- `d[k] += 1` from two handler threads loses one."""
    with _counters:
        for n in names:
            _state[n] += 1
        _state.update(kv)


# --------------------------------------------------------------- model load
def load_backend():
    global N_MELS
    import torch
    import whisper
    import npu_whisper_encoder as E
    import rawxrt

    torch.set_num_threads(TORCH_THREADS)
    torch.set_grad_enabled(False)

    if FAIL_INJECT:
        # Failure-path test instance: never touches the device, so it can run
        # beside the real service without competing for hw_contexts, and skips
        # the 2.5 GB of encoder weights it would never use.
        log("NPU_STT_FAIL_INJECT=1 -- encoder not loaded, every NPU attempt "
            "will fail on purpose")
        return torch, whisper, E, None, None, {}, None, 0, 0

    # The decoder comes first because the encoder weights come out of the same
    # checkpoint: openai-whisper downloads it on the first start and caches it
    # (~/.cache/whisper, or wherever XDG_CACHE_HOME points).
    t0 = time.perf_counter()
    model = whisper.load_model(MODEL).eval()
    N_MELS = model.dims.n_mels
    log(f"{MODEL} checkpoint ready in {time.perf_counter()-t0:.1f} s "
        f"({N_MELS} mel bins, torch threads={TORCH_THREADS})")

    t0 = time.perf_counter()
    if WEIGHTS:
        z = np.load(WEIGHTS)
        W = {k: z[k].astype(np.float32) for k in z.files}
        W.pop("mel", None)
        W.pop("ref_out", None)
    else:
        # state_dict() hands back detached tensors, and the "w." prefix is the
        # same one tools/dump_ref.py writes, so both paths produce one shape.
        W = {"w." + k: v.numpy().astype(np.float32)
             for k, v in model.encoder.state_dict().items()}
    n_layer = 1 + max(int(k.split(".")[2]) for k in W if k.startswith("w.blocks."))
    d_model = W["w.ln_post.weight"].shape[0]
    n_head = d_model // 64
    for k in [k for k in list(W) if k.endswith(".weight") and W[k].ndim == 2
              and not k.endswith("_ln.weight")]:
        W[k[:-len("weight")] + "weight_T"] = np.ascontiguousarray(W[k].T)
    extra = E.prepare_mlp_chunks(W, n_layer, d_model, 0)
    log(f"weights: {n_layer} layers, d={d_model}, {n_head} heads, "
        f"from {WEIGHTS or MODEL + ' checkpoint'}, "
        f"{time.perf_counter()-t0:.1f} s")

    rt = rawxrt.RawRT()
    rt.register_weights([v for v in W.values() if isinstance(v, np.ndarray)] + extra)
    E.gelu = rt.gelu                       # what --npu-gelu does
    log("encoder weights resident on device")

    holder = {}

    class _Stub(torch.nn.Module):
        """Hands the decoder the features the NPU just produced.

        Replacing the module (not just its forward) also drops whisper's own
        encoder weights -- 635M fp32 parameters we would never execute.
        """

        def forward(self, x):
            return holder["feats"]

    model.encoder = _Stub()
    log("torch encoder dropped, decoder ready")

    return torch, whisper, E, rt, model, holder, W, n_layer, n_head


TORCH = WHISPER = E = RT = MODEL_OBJ = HOLDER = W = None
N_LAYER = N_HEAD = 0


def ctx_for(requested):
    """Round the client's audio_ctx up onto the compiled ladder."""
    if not requested:
        requested = DEFAULT_CTX
    requested = max(1, min(LADDER[-1], int(requested)))
    for c in LADDER:
        if requested <= c:
            return c
    return LADDER[-1]


# ------------------------------------------------------------------- audio
def ffmpeg_decode(raw, suffix):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
        f.write(raw)
        f.flush()
        r = subprocess.run(
            [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error",
             "-threads", "1", "-i", f.name,
             "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
            capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise ValueError("ffmpeg: " + r.stderr.decode("utf-8", "replace")[-300:])
    return np.frombuffer(r.stdout, dtype="<f4").copy()


def to_audio(raw, filename):
    if not raw:
        raise ValueError("empty audio")
    if raw[:4] == b"RIFF":
        try:
            return pcm16_mono(raw, SR)[0]
        except Exception:
            pass                            # non-PCM16 or resampling needed
    ext = os.path.splitext(filename or "")[1] or ".bin"
    return ffmpeg_decode(raw, ext)


# --------------------------------------------------------------- multipart
_NAME = re.compile(r'name="([^"]*)"')
_FILENAME = re.compile(r'filename="([^"]*)"')


def parse_multipart(body, boundary):
    fields, files = {}, {}
    for chunk in body.split(b"--" + boundary):
        if chunk[:2] == b"--":
            break
        chunk = chunk.lstrip(b"\r\n")
        head, sep, data = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue
        if data.endswith(b"\r\n"):
            data = data[:-2]
        disp = ""
        for line in head.split(b"\r\n"):
            k, _, v = line.partition(b":")
            if k.strip().lower() == b"content-disposition":
                disp = v.decode("utf-8", "replace")
        m = _NAME.search(disp)
        if not m:
            continue
        fn = _FILENAME.search(disp)
        if fn:
            files[m.group(1)] = (fn.group(1), data)
        else:
            fields[m.group(1)] = data.decode("utf-8", "replace")
    return fields, files


# ------------------------------------------------------------------ engine
def ensure_geometry(ctx):
    """Make ctx the only geometry resident on the device.

    Costs one weight re-upload on the first pass after a switch (+0.6 s
    measured); the alternative is a permanent drop to the host path.
    """
    if _state["ctx"] == ctx:
        return
    if _state["ctx"] is not None:
        # _scratch must go with the contexts. Its ("mlpdev", ...) entry holds
        # sub-BOs carved out of the fc2 design's A buffer; evicting that design
        # frees A, the next pass allocates a new one, and the stale sub-BOs then
        # feed fc2 from memory nothing writes to. That is a SILENT wrong answer
        # -- normal timings, no fallback, garbage text -- which is how it was
        # found: the service returned 'and the other one?' for Russian speech
        # that the same code transcribed correctly before any switch.
        RT._chains.clear()
        RT._scratch.clear()
        RT._scratch_owner.clear()
        while RT._live:
            RT._evict_lru()
        RT._views.clear()
        bump("teardowns", ctx=None)
        if RECOVER and not (RT.chain_ok and RT.mlp_ok and RT.softmax_ok):
            log(f"device paths were off ({RT.fallbacks[-1] if RT.fallbacks else '?'})"
                " -- re-enabling after teardown")
            RT.chain_ok = RT.fused_ok = RT.mlp_ok = RT.softmax_ok = True
    # ctx is recorded by transcribe() once a pass has actually completed. Doing
    # it here would mark a geometry resident that a failed pass never built, and
    # the next request with the same ctx would skip the teardown it needs.


def _window(audio, ctx, language, decode, t):
    """One 30 s whisper window: mel -> NPU encoder -> CPU decoder."""
    t0 = time.perf_counter()
    padded = WHISPER.pad_or_trim(audio)
    mel = WHISPER.log_mel_spectrogram(TORCH.from_numpy(padded), n_mels=N_MELS)
    mel_np = np.ascontiguousarray(mel.numpy().astype(np.float32)[:, :2 * ctx])
    t["mel_ms"] += (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    feats = E.encoder(W, mel_np, RT.matmul, RT.matmul, prof={},
                      npu_softmax=False, n_layer=N_LAYER, n_head=N_HEAD,
                      attn_fn=RT.attn_chain, mlp_fn=RT.mlp, fold_bias=True)
    t["encoder_ms"] += (time.perf_counter() - t0) * 1e3
    if not decode:
        return ""

    t0 = time.perf_counter()
    HOLDER["feats"] = TORCH.from_numpy(feats).unsqueeze(0).float()
    opts = WHISPER.DecodingOptions(fp16=False, language=language,
                                   without_timestamps=True, beam_size=None,
                                   temperature=0.0)
    res = WHISPER.decode(MODEL_OBJ, mel.unsqueeze(0), opts)
    r = res[0] if isinstance(res, list) else res
    HOLDER.pop("feats", None)
    t["decoder_ms"] += (time.perf_counter() - t0) * 1e3
    return r.text


def counters():
    return (f"npu={_state['served_npu']} fb_busy={_state['fb_busy']} "
            f"fb_error={_state['fb_error']} fb_decode={_state['fb_decode']} "
            f"fb_failed={_state['fb_failed']}")


def recover_if_degraded():
    """A device path that fell back stays off for the life of the process.

    That is rawxrt's deliberate policy, and for a one-shot script it is right.
    For a service it means one transient error costs every later request: the
    host attention path is correct but takes 13 s at ctx=1500, which is worse
    than the CPU endpoint. So on the next request, release the geometry (the
    fix that made context switching work at all) and let the device try again.
    Fails again -> that request proxies to the CPU, which is the point.
    """
    if RT is None or (RT.chain_ok and RT.mlp_ok and RT.softmax_ok):
        return False
    log(f"device paths degraded ({RT.fallbacks[-1] if RT.fallbacks else '?'})"
        " -- releasing geometry and re-arming")
    RT._chains.clear()
    RT._scratch.clear()
    RT._scratch_owner.clear()
    while RT._live:
        RT._evict_lru()
    RT._views.clear()
    bump("teardowns", "degradations", ctx=None)
    if RECOVER:
        RT.chain_ok = RT.fused_ok = RT.mlp_ok = RT.softmax_ok = True
        bump(degraded=False)
    return True


def proxy_cpu(body, ctype, reason, detail=""):
    """Hand the client's own request to whisper.cpp and pass the answer back.

    The original bytes are forwarded rather than a rebuilt form, so what the
    client gets is byte-for-byte what it would have got from whisper.cpp itself.
    """
    t0 = time.perf_counter()
    req = urllib.request.Request(CPU_URL, data=body,
                                 headers={"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=CPU_TIMEOUT) as r:
            payload, status = r.read(), r.status
            ct = r.headers.get("Content-Type", "application/json")
    except urllib.error.HTTPError as e:          # it answered, just not 200
        payload, status = e.read(), e.code
        ct = e.headers.get("Content-Type", "application/json")
    dt = (time.perf_counter() - t0) * 1e3
    log(f"FALLBACK->CPU [{reason}] {dt:.0f} ms http={status} {counters()}"
        + (f" :: {detail}" if detail else ""))
    return status, payload, ct, dt


def transcribe(audio, ctx, language, decode=True):
    """Whole recording -> text. Caller holds the device lock.

    Whisper sees 30 s at a time, and recordings are longer than that. Without the
    window loop a 45 s recording came back with only its first half while
    whisper.cpp returned all of it -- silently, which is the worst way to be
    wrong. Windows are a fixed 30 s stride and carry no prompt between them,
    which is what whisper.cpp does too (`-mc 0`); its seek is timestamp-driven,
    so a word straddling a boundary can be split differently here.
    """
    t = {"mel_ms": 0.0, "encoder_ms": 0.0, "decoder_ms": 0.0, "windows": 0}
    if FAIL_INJECT:
        raise RuntimeError("injected NPU failure (NPU_STT_FAIL_INJECT)")
    n_fb = len(RT.fallbacks)
    ensure_geometry(ctx)
    parts = []
    for off in range(0, max(audio.size, 1), N_SAMPLES):
        chunk = audio[off:off + N_SAMPLES]
        if off and chunk.size < SR // 2:
            break                       # a sub-half-second tail is not speech
        if MAX_WINDOWS and t["windows"] >= MAX_WINDOWS:
            t["truncated_s"] = (audio.size - off) / SR
            log(f"audio longer than {MAX_WINDOWS * 30} s: "
                f"{t['truncated_s']:.0f} s dropped (NPU_STT_MAX_WINDOWS)")
            break
        t["windows"] += 1
        parts.append(_window(chunk, ctx, language, decode, t).strip())
    bump(ctx=ctx)
    t["fallbacks"] = RT.fallbacks[n_fb:]
    if t["fallbacks"]:
        bump(degraded=True)
    return " ".join(p for p in parts if p), t


def prewarm():
    """Compile/attach every overlay on the ladder, then warm the torch decoder.

    Encoder only per context: the overlays are what is expensive and what is
    per-context. The decoder is warmed once, on the shortest context.
    """
    silence = np.zeros(SR * 30, dtype=np.float32)
    ok = 0
    for c in LADDER:
        t0 = time.perf_counter()
        try:
            transcribe(silence, c, LANG, decode=False)
            ok += 1
            log(f"prewarm ctx={c}: {(time.perf_counter()-t0)*1e3:.0f} ms")
        except Exception as e:
            log(f"prewarm ctx={c} FAILED: {type(e).__name__}: {e}")
    t0 = time.perf_counter()
    try:
        transcribe(silence[:SR], LADDER[0], LANG)
        log(f"prewarm decoder: {(time.perf_counter()-t0)*1e3:.0f} ms")
    except Exception as e:
        log(f"prewarm decoder FAILED: {type(e).__name__}: {e}")
    # Swallowing these used to leave /health saying "ok" on a device where not a
    # single overlay had compiled: every request then fell back, and the only
    # symptom was the latency.
    if not ok:
        raise RuntimeError(
            f"prewarm failed on every context in the ladder ({LADDER}) -- the "
            f"device is not usable; see the FAILED lines above")
    if ok < len(LADDER):
        bump(degraded=True)
        log(f"prewarm: {ok}/{len(LADDER)} contexts usable, marked degraded")


# -------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "npu-stt/1.0"

    def log_message(self, fmt, *args):        # BaseHTTPRequestHandler noise
        pass

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] in ("/health", "/healthz"):
            self._json(200, {"status": "ok" if _state["ready"] else "loading",
                             "model": MODEL, "ladder": LADDER,
                             "resident_ctx": _state["ctx"],
                             "max_windows": MAX_WINDOWS,
                             "requests": _state["requests"],
                             "errors": _state["errors"],
                             "teardowns": _state["teardowns"],
                             "degraded": _state["degraded"],
                             "degradations": _state["degradations"],
                             "device_paths": {
                                 "chain": bool(RT.chain_ok) if RT else None,
                                 "mlp": bool(RT.mlp_ok) if RT else None},
                             "cpu_fallback": {
                                 "url": CPU_URL or None,
                                 "busy_wait_ms": BUSY_WAIT_MS,
                                 "served_npu": _state["served_npu"],
                                 "busy": _state["fb_busy"],
                                 "error": _state["fb_error"],
                                 "decode": _state["fb_decode"],
                                 "proxy_failed": _state["fb_failed"]},
                             "uptime_s": round(time.time() - _state["started"], 1)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path not in ("/inference", "/v1/audio/transcriptions"):
            return self._json(404, {"error": "not found"})
        if not _state["ready"]:
            return self._json(503, {"error": "model still loading"})

        t_req = time.perf_counter()
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._json(400, {"error": "bad Content-Length"})
        if n <= 0:
            return self._json(400, {"error": "empty body"})
        if n > MAX_BODY:
            return self._json(413, {"error": f"body over {MAX_BODY} bytes"})

        ctype = self.headers.get("Content-Type", "")
        m = re.search(r'boundary="?([^";]+)"?', ctype)
        if "multipart/form-data" not in ctype or not m:
            return self._json(400, {"error": "expected multipart/form-data"})

        buf = bytearray()                    # += on bytes recopies the lot
        while len(buf) < n:                  # read() can come up short
            chunk = self.rfile.read(min(1 << 20, n - len(buf)))
            if not chunk:
                break
            buf += chunk
        if len(buf) != n:
            return self._json(400, {"error": "short body"})
        body = bytes(buf)
        del buf

        try:
            fields, files = parse_multipart(body, m.group(1).encode())
        except Exception as e:
            return self._json(400, {"error": f"bad multipart: {e}"})
        if "file" not in files:
            return self._json(400, {"error": "no 'file' part"})

        filename, raw = files["file"]
        language = (fields.get("language") or LANG).strip() or LANG
        try:
            req_ctx = int(float(fields.get("audio_ctx") or 0))
        except ValueError:
            req_ctx = 0
        ctx = ctx_for(req_ctx)

        # The original body is kept until the answer is out: falling back means
        # replaying exactly these bytes at whisper.cpp.
        def fallback(reason, detail=""):
            bump({"busy": "fb_busy", "error": "fb_error",
                  "decode": "fb_decode"}[reason], "requests")
            try:
                status, payload, ct, dt = proxy_cpu(body, ctype, reason, detail)
            except Exception as exc:
                bump("fb_failed", "errors")
                log(f"FALLBACK->CPU [{reason}] ITSELF FAILED: "
                    f"{type(exc).__name__}: {exc} :: {detail}")
                return self._json(502, {"error": f"npu {reason} ({detail}); "
                                                 f"cpu fallback: {exc}"})
            self.send_response(status)
            self.send_header("Content-Type", ct)
            self.send_header("X-STT-Path", "cpu")
            self.send_header("X-STT-Fallback", reason)
            self.send_header("X-STT-Cpu-Ms", str(round(dt)))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        try:
            audio = to_audio(raw, filename)
            if audio.size < SR // 20:
                raise ValueError("audio shorter than 50 ms")
        except Exception as e:
            # Our ffmpeg is 7.1.5, the whisper.cpp side ran 4.4.2 plus
            # miniaudio: what one refuses the other may take. That side rejects
            # real garbage in ~30 ms, so the client gets its verdict either way.
            if CPU_URL:
                return fallback("decode", f"{type(e).__name__}: {e}")
            bump("errors")
            return self._json(400, {"error": f"cannot decode audio: {e}"})
        del raw

        wait = (BUSY_WAIT_MS / 1000.0) if CPU_URL else QUEUE_WAIT_S
        if not _lock.acquire(timeout=wait):
            if CPU_URL:
                return fallback("busy", f"waited {BUSY_WAIT_MS:.0f} ms")
            bump("errors")
            return self._json(503, {"error": "device busy"})
        err = None
        try:
            recover_if_degraded()
            text, t = transcribe(audio, ctx, language)
        except Exception as e:
            log("NPU PATH FAILED: "
                + traceback.format_exc().strip().replace("\n", " | "))
            err = e
        finally:
            # Released BEFORE the proxy call: the CPU round trip takes seconds
            # and holding the device through it would make one failure block
            # every healthy request behind it.
            _lock.release()
        if err is not None:
            if CPU_URL:
                return fallback("error", f"{type(err).__name__}: {err}")
            bump("errors")
            return self._json(500, {"error": f"{type(err).__name__}: {err}"})

        total = (time.perf_counter() - t_req) * 1e3
        bump("requests", "served_npu")
        del body
        log(f"ok {audio.size/SR:5.1f}s ctx={req_ctx}->{ctx} win={t['windows']} "
            f"mel={t['mel_ms']:.0f} enc={t['encoder_ms']:.0f} "
            f"dec={t['decoder_ms']:.0f} total={total:.0f} ms"
            + (f" FALLBACKS={t['fallbacks']}" if t["fallbacks"] else "")
            + (f" [{counters()}]" if _state["requests"] % 25 == 0 else "")
            + (f" :: {text.strip()[:80]}" if LOG_TEXT else ""))

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("X-STT-Path", "npu")
        if t.get("truncated_s"):
            self.send_header("X-STT-Truncated-S", str(round(t["truncated_s"])))
        for k, v in (("X-NPU-Ctx", ctx), ("X-NPU-Mel-Ms", round(t["mel_ms"])),
                     ("X-NPU-Enc-Ms", round(t["encoder_ms"])),
                     ("X-NPU-Dec-Ms", round(t["decoder_ms"])),
                     ("X-NPU-Total-Ms", round(total))):
            self.send_header(k, str(v))
        payload = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    global TORCH, WHISPER, E, RT, MODEL_OBJ, HOLDER, W, N_LAYER, N_HEAD
    log(f"starting on {HOST}:{PORT}, ladder={LADDER}, default_ctx={DEFAULT_CTX}")
    log(f"cpu fallback: {CPU_URL or 'DISABLED'}"
        + (f", busy_wait={BUSY_WAIT_MS:.0f} ms, timeout={CPU_TIMEOUT:.0f} s"
           if CPU_URL else ""))
    (TORCH, WHISPER, E, RT, MODEL_OBJ, HOLDER, W,
     N_LAYER, N_HEAD) = load_backend()
    if PREWARM:
        prewarm()
    _state["ready"] = True
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    srv.daemon_threads = True
    log(f"ready on {HOST}:{PORT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")


if __name__ == "__main__":
    main()
