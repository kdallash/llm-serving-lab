
# Qwen3-0.6B: Weight Memory and KV Cache Estimation

## 1. Estimate Model Weight Memory

Qwen3-0.6B has approximately 0.6 billion parameters. We can estimate its weight memory using the number of bytes required to store each parameter.

**Formula:**

\[
\text{Weight Memory} = \text{Number of Parameters} \times \text{Bytes per Parameter}
\]

| Precision | Bytes per Parameter | Estimated Memory |
|---|---:|---:|
| FP16 | 2 | 1.2 GB |
| INT8 | 1 | 0.6 GB |
| INT4 | 0.5 | 0.3 GB |

**Note:** These are theoretical estimates using decimal GB. Actual memory usage may be higher because of quantization metadata, scales, and runtime overhead.

## 2. Calculate KV Cache Memory Per Token

The KV cache stores attention keys and values from previously processed tokens. This avoids recomputing them during generation but consumes GPU memory.

Assuming the model configuration contains:

- Number of layers: 28
- Number of KV heads: 8
- Head dimension: 128
- KV cache precision: FP16 (2 bytes per value)

**Formula:**

\[
M_{\text{KV/token}} =
2 \times L \times H_{\text{KV}} \times D \times B
\]

Where:
- \(2\) represents keys and values (K and V).
- \(L\) is the number of transformer layers.
- \(H_{\text{KV}}\) is the number of KV heads.
- \(D\) is the head dimension.
- \(B\) is the number of bytes per value.

**Calculation:**

\[
2 \times 28 \times 8 \times 128 \times 2
= 114{,}688 \text{ bytes}
\]

Therefore, the KV cache requires **112 KiB per token**.

## 3. Estimate KV Cache for 32 Concurrent Requests

Assume each request has 1,024 tokens in its KV cache and all 32 requests are active concurrently.

\[
M_{\text{KV}} =
M_{\text{KV/token}} \times
\text{Tokens per Request} \times
\text{Number of Requests}
\]

\[
= 114{,}688 \times 1{,}024 \times 32
\]

\[
= 3{,}758{,}096{,}384 \text{ bytes}
\]

The total KV cache memory is approximately **3.5 GiB** (3.76 GB in decimal units).

## 4. Estimate Remaining GPU Memory

Assume an NVIDIA T4 with 16 GiB of GPU memory and FP16 model weights.

| Component | Estimated Memory |
|---|---:|
| Total GPU memory | 16 GiB |
| FP16 model weights | 1.12 GiB |
| KV cache for 32 requests | 3.5 GiB |
| Remaining memory before runtime overhead | 11.4 GiB |

\[
16 - 1.12 - 3.5 \approx 11.38 \text{ GiB}
\]

The remaining memory must also accommodate CUDA allocations, activations, temporary workspaces, and inference-engine overhead.

**Important:** The full 16 GiB may not be available to the model. Check actual GPU memory usage with `nvidia-smi`.

## 5. Conclusion

- **Model weights:** FP16 requires approximately 1.2 GB, INT8 requires 0.6 GB, and INT4 requires 0.3 GB before additional overhead.
- **KV cache:** FP16 requires 112 KiB per token under the given configuration.
- **Concurrency:** 32 requests with 1,024 cached tokens each require approximately 3.5 GiB of KV cache.
- **Memory planning:** Subtract both model weights and KV cache from GPU memory, then account for runtime overhead.

These estimates assume the KV cache uses FP16 and every request holds 1,024 cached tokens. Actual usage depends on the inference engine, cache precision, batch size, and sequence lengths.
