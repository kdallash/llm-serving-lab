"""Results schema (v1.0). Every benchmark/eval run saves one RunResult json;
benchmark runs also save one RequestRecord per line (jsonl).

Units: latency in ms, throughput in tokens/s and requests/s, memory in MB,
timestamps in seconds from benchmark start.
"""
import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # typos in field names fail loudly


class Percentiles(Strict):
    p50: float
    p95: float
    p99: float

    @model_validator(mode="after")
    def ordered(self):
        if not (self.p50 <= self.p95 <= self.p99):
            raise ValueError("percentiles must satisfy p50 <= p95 <= p99")
        return self


class WorkloadInfo(Strict):
    name: str
    sha256: str
    num_requests: int
    seed: int
    rate: Optional[float] = None  # req/s for online workloads


class Throughput(Strict):
    output_tokens_per_s: float
    requests_per_s: float


class Accuracy(Strict):
    wikitext_word_perplexity: Optional[float] = None
    task_name: Optional[str] = None
    task_acc: Optional[float] = None


class Environment(Strict):
    git_commit: str
    gpu_name: str
    attention_backend: Optional[str] = None
    versions: Dict[str, str]


class RunResult(Strict):
    schema_version: Literal["1.0"]
    run_id: str
    run_type: Literal["benchmark", "eval", "baseline"]
    variant: Literal["fp16", "int8", "int4"]
    backend: Literal["hf", "vllm"]
    status: Literal["ok", "partial", "failed"]
    # benchmark-only fields (None for eval runs)
    workload: Optional[WorkloadInfo] = None
    concurrency: Optional[int] = None
    throughput: Optional[Throughput] = None
    ttft_ms: Optional[Percentiles] = None
    itl_ms: Optional[Percentiles] = None
    e2e_ms: Optional[Percentiles] = None
    # always present
    peak_gpu_memory_mb: float
    memory_measure_method: str  # e.g. "torch.cuda.max_memory_allocated"
    accuracy: Accuracy = Accuracy()
    environment: Environment
    config: dict            # fully resolved config
    config_sha256: str

    @model_validator(mode="after")
    def benchmark_needs_latency(self):
        if self.run_type == "benchmark" and self.status == "ok":
            for f in ("workload", "concurrency", "throughput", "ttft_ms", "itl_ms", "e2e_ms"):
                if getattr(self, f) is None:
                    raise ValueError(f"benchmark run is missing '{f}'")
        return self


class RequestRecord(Strict):
    id: int
    send_time: float
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    prompt_tokens: int
    output_tokens: int
    token_timestamps: List[float] = []
    error: Optional[str] = None

    @model_validator(mode="after")
    def times_make_sense(self):
        if self.error is None:
            if self.first_token_time is None or self.end_time is None:
                raise ValueError("successful request needs first_token_time and end_time")
            if not (self.send_time <= self.first_token_time <= self.end_time):
                raise ValueError("need send_time <= first_token_time <= end_time")
        return self


def validate_result_file(path) -> RunResult:
    return RunResult.model_validate(json.loads(Path(path).read_text()))


def validate_request_file(path) -> List[RequestRecord]:
    with open(path, encoding="utf-8") as f:
        return [RequestRecord.model_validate(json.loads(l)) for l in f if l.strip()]
