"""Run the evaluation for one variant with lm-evaluation-harness.

    python eval/run_eval.py --config configs/fp16.yaml --out-dir results

Writes into --out-dir (prefix eval_<variant>):
    _raw_results.json   lm-eval output as returned, minus the per-sample logs
    _raw_samples.json   the per-sample logs, untouched (large, no need to commit)
    _samples.jsonl      compact per-sample records
    _result.json        validated result (your schema, run_type "eval")
"""
import argparse
import datetime
import importlib.metadata
import inspect
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # so `bench.*` and `eval.*` import from any directory

import lm_eval
import torch
import transformers

from bench.config import cfg_hash, load_config
from bench.git_helper import git_commit
from bench.schema import RunResult
from eval.eval_core import (build_model_args, extract_metrics, headline_summary,
                            json_default, seed_kwargs, write_samples)

MIB = 1024 ** 2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fp16.yaml")
    ap.add_argument("--out-dir", default="results")
    return ap.parse_args()


def check_signature(names):
    """Fail early and clearly if this lm-eval version lacks a parameter we pass."""
    params = inspect.signature(lm_eval.simple_evaluate).parameters
    missing = [n for n in names if n not in params]
    if missing:
        raise RuntimeError(f"installed lm_eval.simple_evaluate has no parameter(s): {missing}")


def lm_eval_version():
    try:
        return importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"config: {args.config}\nout-dir: {out_dir}")

    assert torch.cuda.is_available(), "No GPU found. Run this on Kaggle, not locally."
    cfg = load_config(args.config)
    ev = cfg["evaluation"]
    assert ev["backend"] == "hf", "this script only drives the HF backend"
    assert ev["log_samples"] is True, "per-sample records need evaluation.log_samples: true"
    print("RESOLVED CONFIG:")
    print(json.dumps(cfg, indent=2))
    c_hash = cfg_hash(cfg)
    print("config hash:", c_hash)

    prefix = f"eval_{cfg['variant']}"
    model_args = build_model_args(cfg)
    print("lm-eval model_args:", model_args)

    kwargs = dict(
        model="hf", model_args=model_args, tasks=ev["tasks"],
        num_fewshot=ev["num_fewshot"], limit=ev["limit"],
        batch_size=ev["batch_size"], device=ev["device"],
        log_samples=ev["log_samples"], **seed_kwargs(cfg),
    )
    check_signature(kwargs)

    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        raw = lm_eval.simple_evaluate(**kwargs)
        elapsed = time.time() - t0
        peak_mib = torch.cuda.max_memory_allocated() / MIB
        print(f"evaluation finished in {elapsed:.1f} s ({elapsed / 60:.1f} min)")
        assert raw is not None, "simple_evaluate returned nothing"

        # STEP 6: save raw output first, before parsing anything
        samples = raw.pop("samples", None)
        raw_path = out_dir / f"{prefix}_raw_results.json"
        raw_path.write_text(json.dumps(raw, indent=2, default=json_default))
        print("saved", raw_path)
        if samples is None:
            raise RuntimeError("no 'samples' in lm-eval output")
        (out_dir / f"{prefix}_raw_samples.json").write_text(
            json.dumps(samples, default=json_default))

        # STEP 7: parse (fails loudly if anything is missing)
        extracted = extract_metrics(raw, ev)

        # STEP 9: per-sample records
        n_rec, n_without, s_hash = write_samples(out_dir / f"{prefix}_samples.jsonl", samples, ev)
        print(f"samples: {n_rec} records, {n_without} without reported metrics, sha256 {s_hash[:12]}")

        # STEP 8: build + validate (constructing RunResult validates it)
        result = RunResult(
            schema_version="1.0",
            run_id=f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d}_{cfg['variant']}_eval",
            run_type="eval", variant=cfg["variant"], backend="hf", status="ok",
            peak_gpu_memory_mb=peak_mib,
            memory_measure_method="torch.cuda.max_memory_allocated, reported in MiB",
            accuracy={"task_metrics": extracted},
            environment={
                "git_commit": git_commit(),
                "gpu_name": torch.cuda.get_device_name(0),
                "versions": {"torch": torch.__version__,
                             "transformers": transformers.__version__,
                             "lm_eval": lm_eval_version()},
            },
            config=cfg, config_sha256=c_hash,
            extra={
                "elapsed_s": elapsed, "tasks": ev["tasks"], "batch_size": ev["batch_size"],
                "max_length": ev["max_length"], "num_fewshot": ev["num_fewshot"],
                "limit": ev["limit"], "seeds": ev["seeds"],
                "n_samples_logged": n_rec, "samples_without_metrics": n_without,
                "samples_sha256": s_hash, "raw_results_file": raw_path.name,
            },
        )
        res_path = out_dir / f"{prefix}_result.json"
        res_path.write_text(result.model_dump_json(indent=2))
    except Exception as e:
        # a failed run should be visible, not silently missing
        (out_dir / f"{prefix}_failed.json").write_text(json.dumps(
            {"status": "failed", "error": repr(e), "git_commit": git_commit(),
             "config_sha256": c_hash}, indent=2))
        raise

    # STEP 10: summary for the Kaggle log
    print("\n=== SUMMARY ===")
    for task, (metric, value, err) in headline_summary(extracted, ev).items():
        print(f"{task:15s} {metric:16s} {value:.4f}" + (f" +/- {err:.4f}" if err is not None else ""))
    print("commit:", result.environment.git_commit, "| config hash:", c_hash[:12])
    print("peak GPU memory (MiB):", round(peak_mib, 1))
    print("saved", res_path)


if __name__ == "__main__":
    main()
