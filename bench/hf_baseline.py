"""HF transformers fp16 baseline: one request, warmup, tokens/s, peak VRAM.

Run on a GPU (Kaggle):
    python bench/hf_baseline.py --config configs/fp16.yaml --out /kaggle/working/hf_baseline_fp16.json
"""
import argparse
import datetime
import statistics
import subprocess
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import cfg_hash, load_config
from schema import RunResult

MIB = 1024 ** 2
FIXED_TEXT = ("The quick brown fox jumps over the lazy dog. "
              "Quantization reduces the memory needed to store model weights. ")


def git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def timed_generate(model, ids, n_new):
    """Greedy generation of exactly n_new tokens (min_new_tokens blocks early EOS)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                         max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, out.shape[1] - ids.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fp16.yaml")
    ap.add_argument("--out", default="results/hf_baseline_fp16.json")
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()

    assert torch.cuda.is_available(), "No GPU found. Run this on Kaggle, not locally."
    cfg = load_config(a.config)
    assert cfg["variant"] == "fp16", "HF baseline is fp16 only"
    print("RESOLVED CONFIG:", cfg)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float16)
    model = model.to("cuda").eval()
    weights_mib = torch.cuda.memory_allocated() / MIB

    mc = model.config
    head_dim = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)
    kv_bytes_per_token = 2 * mc.num_hidden_layers * mc.num_key_value_heads * head_dim * 2
    print(f"layers={mc.num_hidden_layers} kv_heads={mc.num_key_value_heads} "
          f"head_dim={head_dim} -> KV/token={kv_bytes_per_token} bytes")
    print(f"weights on GPU: {weights_mib:.1f} MiB")

    ids = tok(FIXED_TEXT * 200, return_tensors="pt").input_ids[:, :a.prompt_tokens].to("cuda")
    assert ids.shape[1] == a.prompt_tokens

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(2):                       # warmup
            timed_generate(model, ids, 16)
        ttfts, fulls = [], []
        for _ in range(a.repeats):
            t1, _ = timed_generate(model, ids, 1)            # ~ prefill + 1 token = TTFT
            tf, n = timed_generate(model, ids, a.new_tokens)
            assert n == a.new_tokens
            ttfts.append(t1)
            fulls.append(tf)
    peak_mib = torch.cuda.max_memory_allocated() / MIB

    ttft, full = statistics.median(ttfts), statistics.median(fulls)
    decode_tps = (a.new_tokens - 1) / max(full - ttft, 1e-9)
    print(f"TTFT {ttft*1000:.1f} ms | total {full:.2f} s | "
          f"{a.new_tokens/full:.1f} tok/s overall | {decode_tps:.1f} tok/s decode | peak {peak_mib:.0f} MiB")

    result = RunResult(
        schema_version="1.0",
        run_id=f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d}_{cfg['variant']}_hf_baseline",
        run_type="baseline", variant=cfg["variant"], backend="hf", status="ok",
        throughput={"output_tokens_per_s": a.new_tokens / full, "requests_per_s": 1 / full},
        peak_gpu_memory_mb=peak_mib,
        memory_measure_method="torch.cuda.max_memory_allocated, reported in MiB",
        environment={
            "git_commit": git_commit(),
            "gpu_name": torch.cuda.get_device_name(0),
            "attention_backend": getattr(mc, "_attn_implementation", None),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        },
        config=cfg, config_sha256=cfg_hash(cfg),
        extra={
            "weights_mib": weights_mib, "kv_bytes_per_token": kv_bytes_per_token,
            "prompt_tokens": a.prompt_tokens, "new_tokens": a.new_tokens,
            "repeats": a.repeats, "ttft_ms_median": ttft * 1000,
            "decode_tokens_per_s": decode_tps,
        },
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(result.model_dump_json(indent=2))
    print("saved", a.out)


if __name__ == "__main__":
    main()
