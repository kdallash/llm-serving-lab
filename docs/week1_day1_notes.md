# Week 1, Day 1: memory arithmetic and HF baseline

Model: Qwen3-0.6B. Run: `results/hf_baseline_fp16.json` (run_id `2026-10-22_fp16_hf_baseline`).

## Setup of the measured run

- GPU: Tesla T4 (one), Kaggle batch run
- Commit: `d3837ea4ddd60eb21e7a4ccdb8d8f49b0fe22ff2` (not dirty)
- torch 2.11.0+cu128, transformers 5.16.1, attention backend: sdpa
- HF transformers, fp16, single request, greedy, 128 prompt tokens, exactly 256 new tokens
- 2 warmup generations, then 5 repeats, medians reported

## 1. Prediction: weight memory

Formula: parameters x bytes per parameter. Qwen3-0.6B has about 0.6B parameters.

| Precision | Bytes per param | Predicted |
|---|---:|---:|
| fp16 | 2 | 1.2 GB (about 1,144 MiB) |
| int8 | 1 | 0.6 GB |
| int4 | 0.5 | 0.3 GB |

Refinement (my own arithmetic, to be checked in Week 2): the embedding matrix (about 151,936 x 1,024, roughly 0.16B parameters, tied with `lm_head`) is expected to stay fp16 because the quantization recipes skip it. That is about 0.31 GB that does not shrink, so the quantized models should be larger than the naive numbers:

| Precision | Refined estimate |
|---|---:|
| int8 (W8A8) | about 0.75 GB |
| int4 (W4A16, group scales included) | about 0.55 GB |

## 2. Prediction: KV cache

Values read from the model config: 28 layers, 8 KV heads, head dim 128, fp16 cache.

KV per token = 2 (K and V) x 28 x 8 x 128 x 2 bytes = **114,688 bytes (112 KiB)**.

For 32 concurrent requests with 1,024 tokens each: 114,688 x 1,024 x 32 = 3,758,096,384 bytes, about **3.5 GiB**. On a 16 GB T4 that is large next to a 1.1 GiB model, so at higher concurrency the KV cache, not the weights, is what fills memory.

## 3. Measured

| | Predicted | Measured |
|---|---:|---:|
| Weights on GPU | about 1,144 MiB | 1,136.9 MiB |
| KV per token | 114,688 bytes | 114,688 bytes (printed from config) |
| Peak allocated | not predicted | 1,200.6 MiB |

- Weights are within about 0.7% of the prediction. The parameter count implied by the measurement is about 0.596B, and the tied embedding is stored once.
- Peak minus weights is about 64 MiB. The KV cache for this run (384 tokens x 114,688 bytes) is about 42 MiB. The remaining roughly 22 MiB is activations, logits and temporary buffers. I did not separate those further.

## 4. Why real VRAM is higher than the weights alone

- KV cache grows with every token held (about 112 KiB per token here).
- Activations and temporary buffers during the forward pass.
- Not measured in this run: `torch.cuda.max_memory_allocated` counts only what PyTorch allocated. `nvidia-smi` would also include the CUDA context and the allocator's cached blocks, so it should read higher. Worth checking once with `nvidia-smi` to see by how much.

## 5. Speed observation

| Metric | Measured |
|---|---:|
| TTFT (128-token prompt, median) | 46.0 ms |
| Decode speed | 25.0 tok/s (about 40 ms per token) |

Expected upper bound if decoding were purely memory-bound: a T4 has roughly 320 GB/s of bandwidth, and each token needs to read about 1.19 GB of weights, so about 3.7 ms per token, or roughly 270 tok/s. The measured speed is about ten times lower.

Hypothesis (not tested): HF eager mode is limited by per-kernel launch and Python overhead across 28 layers, so the GPU is idle much of the time. Supporting hint: a 128-token prefill (46 ms TTFT) costs about the same as a single decode step (about 40 ms), which suggests each forward pass has a large fixed cost that does not depend on the amount of work.

Consequence to check in Week 2: vLLM uses CUDA graphs and fused kernels, so fp16 should be much faster there. For a model this small, int4 may show little speedup if overhead rather than weight reads is the bottleneck. On T4 there is also the known lack of an optimized INT4 kernel.

## 6. Caveats

- Single request, batch size 1. Nothing here says anything about throughput under concurrency.
- Peak memory is the PyTorch allocator number, not `nvidia-smi`.
- Five repeats, medians only. No percentiles, so `ttft_ms`, `itl_ms` and `e2e_ms` are null in the json for this baseline run.
- The Kaggle notebook used `GIT_REF = "main"`. For reproducibility use a commit hash. The committed hash is recorded in the json.
- `requirements-bench.txt` was not found at `/tmp/repo` in the run, so versions are not pinned yet. Pin from the versions recorded above.

## 7. Open questions for Week 2

- Do the int8 and int4 model sizes on disk match the refined estimates (about 0.75 and 0.55 GB)?
- How much faster is vLLM fp16 than this 25 tok/s baseline, and is the gap explained by launch overhead?
- What does `nvidia-smi` show for the same run?
