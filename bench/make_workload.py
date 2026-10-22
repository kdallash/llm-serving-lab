"""Build the seeded offline workload.

Usage:
    python bench/make_workload.py --seed 42 --n 200 --out bench/workloads/offline_v1.jsonl

Same seed + same settings -> byte-identical file (check the .sha256 sidecar).
"""
import argparse
import hashlib
import json
import random
from pathlib import Path


def build_offline(ids, decode, encode, seed, n,
                  prompt_range=(100, 900), out_range=(100, 900)):
    """Pure function: no downloads, so it is easy to unit test.

    ids    : long list of token ids from one fixed source text
    decode : ids -> str
    encode : str -> ids (used only to record the real token count)
    """
    rng = random.Random(seed)  # own RNG, never the global one
    reqs = []
    for i in range(n):
        p_len = rng.randint(*prompt_range)
        o_len = rng.randint(*out_range)
        start = rng.randint(0, len(ids) - p_len - 1)
        prompt = decode(ids[start:start + p_len])
        reqs.append({
            "id": i,
            "prompt": prompt,
            "prompt_tokens": p_len,                       # target
            "prompt_tokens_actual": len(encode(prompt)),  # after decode/re-encode
            "max_tokens": o_len,
            "ignore_eos": True,  # force exactly max_tokens of generation
        })
    return reqs


def write_jsonl(reqs, path):
    """Deterministic serialization, returns sha256 of the file bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in reqs]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    Path(str(path) + ".sha256").write_text(digest + "\n")
    return digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--prompt-min", type=int, default=100)
    ap.add_argument("--prompt-max", type=int, default=900)
    ap.add_argument("--out-min", type=int, default=100)
    ap.add_argument("--out-max", type=int, default=900)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", default="bench/workloads/offline_v1.jsonl")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n".join(ds["text"][:20000])  # fixed slice = reproducible
    tok = AutoTokenizer.from_pretrained(a.model)  # tokenizer only
    ids = tok(text)["input_ids"]

    reqs = build_offline(
        ids,
        decode=tok.decode,
        encode=lambda s: tok(s)["input_ids"],
        seed=a.seed, n=a.n,
        prompt_range=(a.prompt_min, a.prompt_max),
        out_range=(a.out_min, a.out_max),
    )
    digest = write_jsonl(reqs, a.out)
    print(f"wrote {len(reqs)} requests to {a.out}\nsha256 {digest}")


if __name__ == "__main__":
    main()
