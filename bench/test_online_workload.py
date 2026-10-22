"""Run with: pytest tests/test_online_workload.py
No model, no internet.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from make_online_workload import add_poisson_arrivals  # noqa: E402
from make_workload import write_jsonl  # noqa: E402


def base(n):
    return [{"id": i, "prompt": f"p{i}", "prompt_tokens": 100, "max_tokens": 100}
            for i in range(n)]


def gaps(reqs):
    times = [0.0] + [r["arrival_time"] for r in reqs]
    return [b - a for a, b in zip(times, times[1:])]


def test_same_seed_and_rate_same_schedule():
    assert add_poisson_arrivals(base(100), 4, 42) == add_poisson_arrivals(base(100), 4, 42)


def test_arrivals_are_sorted():
    times = [r["arrival_time"] for r in add_poisson_arrivals(base(500), 4, 42)]
    assert times == sorted(times)


def test_mean_gap_close_to_one_over_rate():
    for rate in (1, 4, 16):
        g = gaps(add_poisson_arrivals(base(5000), rate, 42))
        mean = sum(g) / len(g)
        assert abs(mean - 1 / rate) / (1 / rate) < 0.05


def test_changing_rate_does_not_change_prompts():
    a = add_poisson_arrivals(base(50), 2, 42)
    b = add_poisson_arrivals(base(50), 8, 42)
    strip = lambda rs: [{k: v for k, v in r.items() if k != "arrival_time"} for r in rs]
    assert strip(a) == strip(b)


def test_input_not_mutated():
    reqs = base(10)
    add_poisson_arrivals(reqs, 4, 42)
    assert all("arrival_time" not in r for r in reqs)


def test_file_is_byte_identical(tmp_path):
    h1 = write_jsonl(add_poisson_arrivals(base(50), 4, 42), tmp_path / "a.jsonl")
    h2 = write_jsonl(add_poisson_arrivals(base(50), 4, 42), tmp_path / "b.jsonl")
    assert h1 == h2
