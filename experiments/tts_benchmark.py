"""
TTS Output Generation Speed Benchmark
======================================
Compares Soprano vs KittenTTS across multiple test sentences.
Supports Apple Silicon (M1/M2/M3), NVIDIA CUDA, and CPU.

Apple Silicon note
------------------
Some Soprano ops (e.g. aten::unfold_backward) are not yet implemented for MPS.
This script sets PYTORCH_ENABLE_MPS_FALLBACK=1 automatically so those ops
silently fall back to CPU while the rest of the model still runs on MPS.
If that still fails, it retries with full CPU mode.

Requirements
------------
Soprano:
    pip install soprano-tts
    pip install torch torchvision torchaudio   # includes MPS for Apple Silicon

KittenTTS:
    pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl

Optional:
    pip install soundfile tabulate
"""

# ── MUST be set before torch is imported anywhere ─────────────────────────────
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import time
import statistics
import argparse


# ── device auto-detection ─────────────────────────────────────────────────────

def detect_best_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on hardware availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ── helpers ───────────────────────────────────────────────────────────────────

def audio_duration_seconds(audio, sample_rate: int) -> float:
    try:
        return audio.shape[-1] / sample_rate
    except Exception:
        return len(audio) / sample_rate


def fmt(val, digits=3):
    return f"{val:.{digits}f}" if val is not None else "N/A"


# ── test sentences ────────────────────────────────────────────────────────────

TEST_SENTENCES = [
    "Hello, this is a quick speed test.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    (
        "Artificial intelligence is transforming the way we interact with "
        "technology, enabling faster and more natural conversations between "
        "humans and machines every single day."
    ),
    (
        "In the beginning there was silence, and then a voice broke through "
        "the void, carrying words of warmth and wisdom across the digital "
        "expanse, reaching ears that had never heard such clarity before in "
        "the history of synthetic speech synthesis technology."
    ),
]

SOPRANO_SAMPLE_RATE   = 32_000
KITTENTTS_SAMPLE_RATE = 24_000


# ── Soprano model loader (with MPS → CPU fallback) ────────────────────────────

def _load_soprano(device: str, backend: str,
                  cache_size_mb: int, decoder_batch_size: int):
    """
    Try loading SopranoTTS on `device`.
    If it fails on MPS, automatically retry on CPU.
    Returns (model, actual_device) or raises.
    """
    from soprano import SopranoTTS

    try:
        model = SopranoTTS(
            backend=backend,
            device=device,
            cache_size_mb=cache_size_mb,
            decoder_batch_size=decoder_batch_size,
        )
        return model, device
    except Exception as exc:
        # MPS op not implemented — retry on CPU
        if device == "mps":
            print(f"\n  ⚠  MPS init failed ({exc})")
            print("     Retrying on CPU …", end=" ", flush=True)
            model = SopranoTTS(
                backend=backend,
                device="cpu",
                cache_size_mb=cache_size_mb,
                decoder_batch_size=decoder_batch_size,
            )
            return model, "cpu"
        raise


# ── Soprano benchmark ─────────────────────────────────────────────────────────

def bench_soprano(sentences, n_runs=3, device="auto",
                  backend="auto", cache_size_mb=10, decoder_batch_size=1):

    if device == "auto":
        device = detect_best_device()

    # LMDeploy is CUDA-only; always use transformers on MPS / CPU
    if device in ("mps", "cpu") and backend == "auto":
        backend = "transformers"

    print("\n" + "═" * 62)
    print(f"  SOPRANO  (requested device={device}, backend={backend})")
    print("═" * 62)

    try:
        from soprano import SopranoTTS  # noqa: F401 — just check it's installed
    except ImportError:
        print("  ✗ soprano-tts not installed.  Run:  pip install soprano-tts")
        return None

    print("  Loading model …", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        model, actual_device = _load_soprano(
            device, backend, cache_size_mb, decoder_batch_size
        )
    except Exception as e:
        print(f"\n  ✗ Failed to load Soprano: {e}")
        return None
    load_time = time.perf_counter() - t0

    if actual_device != device:
        print(f"done in {load_time:.2f}s  (fell back to {actual_device})")
    else:
        print(f"done in {load_time:.2f}s")

    results = []
    for i, sentence in enumerate(sentences, 1):
        print(f"  Sentence {i}/{len(sentences)} …", end=" ", flush=True)
        gen_times, rtfs, cps_list = [], [], []
        audio_dur = None

        for _ in range(n_runs):
            t0 = time.perf_counter()
            audio = model.infer(sentence)
            elapsed = time.perf_counter() - t0

            audio_dur = audio_duration_seconds(audio, SOPRANO_SAMPLE_RATE)
            gen_times.append(elapsed)
            rtfs.append(audio_dur / elapsed)
            cps_list.append(len(sentence) / elapsed)

        mean_rtf = statistics.mean(rtfs)
        print(f"RTF {mean_rtf:.1f}×")
        results.append({
            "text":          (sentence[:52] + "…") if len(sentence) > 52 else sentence,
            "chars":         len(sentence),
            "audio_dur":     audio_dur,
            "gen_time_mean": statistics.mean(gen_times),
            "gen_time_min":  min(gen_times),
            "rtf_mean":      mean_rtf,
            "rtf_max":       max(rtfs),
            "cps_mean":      statistics.mean(cps_list),
        })

    # streaming first-chunk latency
    print("  Measuring first-chunk latency …", end=" ", flush=True)
    stream_latency_ms = None
    try:
        t0 = time.perf_counter()
        for _ in model.infer_stream(sentences[1], chunk_size=1):
            stream_latency_ms = (time.perf_counter() - t0) * 1000
            break
        print(f"{stream_latency_ms:.1f} ms")
    except Exception as e:
        print(f"N/A ({e})")

    _print_results("Soprano", results, load_time, stream_latency_ms, n_runs)
    return results


# ── KittenTTS benchmark ───────────────────────────────────────────────────────

def bench_kittentts(sentences, n_runs=3,
                    model_id="KittenML/kitten-tts-nano-0.2",
                    voice="expr-voice-2-f"):
    print("\n" + "═" * 62)
    print("  KITTENTTS  (CPU)")
    print("═" * 62)

    try:
        from kittentts import KittenTTS
    except ImportError:
        print("  ✗ kittentts not installed.  Run:")
        print("    pip install https://github.com/KittenML/KittenTTS/"
              "releases/download/0.1/kittentts-0.1.0-py3-none-any.whl")
        return None

    print("  Loading model …", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        model = KittenTTS(model_id)
    except Exception as e:
        print(f"\n  ✗ Failed to load KittenTTS: {e}")
        return None
    load_time = time.perf_counter() - t0
    print(f"done in {load_time:.2f}s")

    results = []
    for i, sentence in enumerate(sentences, 1):
        print(f"  Sentence {i}/{len(sentences)} …", end=" ", flush=True)
        gen_times, rtfs, cps_list = [], [], []
        audio_dur = None

        for _ in range(n_runs):
            t0 = time.perf_counter()
            audio = model.generate(sentence, voice=voice)
            elapsed = time.perf_counter() - t0

            audio_dur = audio_duration_seconds(audio, KITTENTTS_SAMPLE_RATE)
            gen_times.append(elapsed)
            rtfs.append(audio_dur / elapsed)
            cps_list.append(len(sentence) / elapsed)

        mean_rtf = statistics.mean(rtfs)
        print(f"RTF {mean_rtf:.1f}×")
        results.append({
            "text":          (sentence[:52] + "…") if len(sentence) > 52 else sentence,
            "chars":         len(sentence),
            "audio_dur":     audio_dur,
            "gen_time_mean": statistics.mean(gen_times),
            "gen_time_min":  min(gen_times),
            "rtf_mean":      mean_rtf,
            "rtf_max":       max(rtfs),
            "cps_mean":      statistics.mean(cps_list),
        })

    _print_results("KittenTTS", results, load_time, n_runs=n_runs)
    return results


# ── shared results printer ────────────────────────────────────────────────────

def _print_results(name, results, load_time, stream_latency_ms=None, n_runs=3):
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False

    print(f"\n  ── {name} Results (mean of {n_runs} runs) ──\n")

    headers = ["Text snippet", "Chars", "Audio(s)", "GenTime(s)", "RTF↑", "Chars/s"]
    rows = [
        [
            r["text"],
            r["chars"],
            fmt(r["audio_dur"], 2),
            fmt(r["gen_time_mean"], 3),
            fmt(r["rtf_mean"], 1) + "×",
            fmt(r["cps_mean"], 1),
        ]
        for r in results
    ]

    if use_tabulate:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        col_w = [max(len(str(row[i])) for row in rows + [headers]) + 2
                 for i in range(len(headers))]
        sep = "  " + "-" * (sum(col_w) + 2 * len(col_w))
        def row_str(row):
            return "  " + "  ".join(str(v).ljust(col_w[i]) for i, v in enumerate(row))
        print(row_str(headers))
        print(sep)
        for row in rows:
            print(row_str(row))

    print(f"\n  Model load time     : {load_time:.2f}s")
    if stream_latency_ms is not None:
        print(f"  First-chunk latency : {stream_latency_ms:.1f} ms")
    all_rtf = [r["rtf_mean"] for r in results]
    best    = max(r["rtf_max"] for r in results)
    print(f"  Overall mean RTF    : {statistics.mean(all_rtf):.1f}×  "
          f"(best single run: {best:.1f}×)")


# ── head-to-head summary ──────────────────────────────────────────────────────

def print_comparison(soprano_results, kitten_results):
    if not soprano_results or not kitten_results:
        return

    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False

    print("\n" + "═" * 62)
    print("  HEAD-TO-HEAD  RTF  (higher = faster)")
    print("═" * 62 + "\n")

    headers = ["Text snippet", "Soprano RTF", "KittenTTS RTF", "Winner"]
    rows = []
    for s, k in zip(soprano_results, kitten_results):
        sr, kr = s["rtf_mean"], k["rtf_mean"]
        rows.append([
            s["text"],
            fmt(sr, 1) + "×",
            fmt(kr, 1) + "×",
            "🎤 Soprano" if sr >= kr else "🐱 KittenTTS",
        ])

    if use_tabulate:
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    else:
        for r in rows:
            print("  ", "  |  ".join(str(c) for c in r))

    s_avg = statistics.mean(r["rtf_mean"] for r in soprano_results)
    k_avg = statistics.mean(r["rtf_mean"] for r in kitten_results)
    ratio = max(s_avg, k_avg) / max(min(s_avg, k_avg), 0.001)
    faster = "Soprano" if s_avg >= k_avg else "KittenTTS"
    print(f"\n  Overall: {faster} is {ratio:.1f}× faster on average")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark Soprano vs KittenTTS output generation speed."
    )
    p.add_argument("--runs",            type=int, default=3)
    p.add_argument("--skip-soprano",    action="store_true")
    p.add_argument("--skip-kitten",     action="store_true")
    p.add_argument("--soprano-device",  default="auto",
                   help="auto | cuda | mps | cpu  (default: auto)")
    p.add_argument("--soprano-backend", default="auto",
                   choices=["auto", "lmdeploy", "transformers"])
    p.add_argument("--kitten-voice",    default="expr-voice-2-f")
    p.add_argument("--kitten-model",    default="KittenML/kitten-tts-nano-0.2")
    p.add_argument("--save-samples",    action="store_true")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = detect_best_device()

    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║       TTS Speed Benchmark  ·  Soprano vs KittenTTS        ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"  Detected hardware   : {device.upper()}")
    print(f"  MPS fallback env    : PYTORCH_ENABLE_MPS_FALLBACK=1 (auto-set)")
    print(f"  Sentences           : {len(TEST_SENTENCES)}")
    print(f"  Runs / sentence     : {args.runs}")

    soprano_results = None
    kitten_results  = None

    if not args.skip_soprano:
        soprano_results = bench_soprano(
            TEST_SENTENCES,
            n_runs=args.runs,
            device=args.soprano_device,
            backend=args.soprano_backend,
        )

    if not args.skip_kitten:
        kitten_results = bench_kittentts(
            TEST_SENTENCES,
            n_runs=args.runs,
            model_id=args.kitten_model,
            voice=args.kitten_voice,
        )

    print_comparison(soprano_results, kitten_results)

    if args.save_samples:
        _save_samples(device)

    print("\n  Benchmark complete.\n")


def _save_samples(device):
    text = "Benchmark complete. This is a sample output."
    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        print("  (pip install soundfile to save .wav samples)")
        return

    try:
        backend = "transformers" if device in ("mps", "cpu") else "auto"
        model, actual_device = _load_soprano(device, backend, 10, 1)
        model.infer(text, "soprano_sample.wav")
        print(f"  Saved soprano_sample.wav  (device={actual_device})")
    except Exception as e:
        print(f"  Soprano sample failed: {e}")

    try:
        from kittentts import KittenTTS
        m = KittenTTS("KittenML/kitten-tts-nano-0.2")
        audio = m.generate(text, voice="expr-voice-2-f")
        sf.write("kittentts_sample.wav", np.asarray(audio), KITTENTTS_SAMPLE_RATE)
        print("  Saved kittentts_sample.wav")
    except Exception as e:
        print(f"  KittenTTS sample failed: {e}")


if __name__ == "__main__":
    main()