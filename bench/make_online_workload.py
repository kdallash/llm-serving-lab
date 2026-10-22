"""Build seeded online workloads (Poisson arrivals) from the offline workload.

Usage:
    python bench/make_online_workload.py --rates 1 2 4 8

Reads  bench/workloads/offline_v1.jsonl
Writes bench/workloads/online_v1_rate<R>.jsonl (+ .sha256), one file per rate.
Prompts and lengths are copied unchanged; only "arrival_time" (seconds from
benchmark start) is added.
"""
import argparse
import json
import random
from pathlib import Path

from make_workload import write_jsonl


def add_poisson_arrivals(reqs, rate, seed):
    """Return copies of reqs with arrival_time added.

    rate: average requests per second (lambda). Gaps are exponential with
    mean 1/rate.

    The RNG is separate from the one used for prompts and lengths. It always
    draws unit-rate gaps and divides by rate, so the same seed gives the same
    burst pattern at every rate, just stretched or squeezed in time.
    """
    rng = random.Random(f"arrivals-{seed}")
    t = 0.0
    out = []
    for r in reqs:
        t += rng.expovariate(1.0) / rate
        out.append({**r, "arrival_time": round(t, 4)})
    return out


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", default="bench/workloads/offline_v1.jsonl")
    ap.add_argument("--rates", type=float, nargs="+", required=True,
                    help="requests/second, e.g. 1 2 4 8")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="bench/workloads")
    a = ap.parse_args()

    base = read_jsonl(a.offline)
    for rate in a.rates:
        reqs = add_poisson_arrivals(base, rate, a.seed)
        path = Path(a.out_dir) / f"online_v1_rate{rate:g}.jsonl"
        digest = write_jsonl(reqs, path)
        print(f"{path}  n={len(reqs)}  duration={reqs[-1]['arrival_time']}s  sha256={digest[:12]}")


if __name__ == "__main__":
    main()
