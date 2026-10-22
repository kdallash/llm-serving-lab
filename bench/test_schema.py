import copy
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from schema import RequestRecord, RunResult, validate_request_file, validate_result_file  # noqa: E402

P = {"p50": 1, "p95": 2, "p99": 3}
SAMPLE = {
    "schema_version": "1.0", "run_id": "t1", "run_type": "benchmark",
    "variant": "fp16", "backend": "vllm", "status": "ok",
    "workload": {"name": "offline_v1", "sha256": "ab", "num_requests": 200, "seed": 42},
    "concurrency": 32,
    "throughput": {"output_tokens_per_s": 100.0, "requests_per_s": 1.0},
    "ttft_ms": P, "itl_ms": P, "e2e_ms": P,
    "peak_gpu_memory_mb": 1000.0, "memory_measure_method": "nvidia-smi",
    "accuracy": {},
    "environment": {"git_commit": "abc", "gpu_name": "Tesla T4", "versions": {"vllm": "x"}},
    "config": {"dtype": "half"}, "config_sha256": "cd",
}


def test_valid_sample_accepted():
    RunResult.model_validate(SAMPLE)


@pytest.mark.parametrize("field", ["variant", "environment", "peak_gpu_memory_mb", "config_sha256"])
def test_missing_field_rejected(field):
    bad = copy.deepcopy(SAMPLE)
    del bad[field]
    with pytest.raises(ValidationError):
        RunResult.model_validate(bad)


def test_unknown_field_rejected():
    bad = {**SAMPLE, "throughtput": {}}
    with pytest.raises(ValidationError):
        RunResult.model_validate(bad)


def test_bad_percentile_order_rejected():
    bad = copy.deepcopy(SAMPLE)
    bad["ttft_ms"] = {"p50": 5, "p95": 2, "p99": 3}
    with pytest.raises(ValidationError):
        RunResult.model_validate(bad)


def test_benchmark_without_latency_rejected():
    bad = copy.deepcopy(SAMPLE)
    del bad["ttft_ms"]
    with pytest.raises(ValidationError):
        RunResult.model_validate(bad)


def test_eval_run_without_latency_ok():
    ev = copy.deepcopy(SAMPLE)
    for k in ("workload", "concurrency", "throughput", "ttft_ms", "itl_ms", "e2e_ms"):
        del ev[k]
    ev["run_type"] = "eval"
    ev["accuracy"] = {"task_name": "x", "task_acc": 0.5}
    RunResult.model_validate(ev)


def test_request_record_rules():
    ok = {"id": 1, "send_time": 1.0, "first_token_time": 1.5, "end_time": 3.0,
          "prompt_tokens": 100, "output_tokens": 50}
    RequestRecord.model_validate(ok)
    with pytest.raises(ValidationError):
        RequestRecord.model_validate({**ok, "first_token_time": 0.5})


def test_file_helpers(tmp_path):
    (tmp_path / "r.json").write_text(json.dumps(SAMPLE))
    validate_result_file(tmp_path / "r.json")
    rec = {"id": 1, "send_time": 1.0, "first_token_time": 1.5, "end_time": 3.0,
           "prompt_tokens": 100, "output_tokens": 50}
    (tmp_path / "q.jsonl").write_text(json.dumps(rec) + "\n")
    assert len(validate_request_file(tmp_path / "q.jsonl")) == 1
