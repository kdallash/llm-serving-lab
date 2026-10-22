"""Run with: pytest tests/test_workload.py
Needs no model and no internet: uses a fake tokenizer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from make_workload import build_offline, write_jsonl  # noqa: E402

IDS = list(range(50_000))                         # fake token ids
decode = lambda ids: " ".join(map(str, ids))      # fake decode
encode = lambda s: s.split()                      # fake encode (same count)


def build(seed=42, n=50):
    return build_offline(IDS, decode, encode, seed, n)


def test_same_seed_same_workload():
    assert build(42) == build(42)


def test_different_seed_differs():
    assert build(42) != build(43)


def test_lengths_within_bounds():
    for r in build():
        assert 100 <= r["prompt_tokens"] <= 900
        assert 100 <= r["max_tokens"] <= 900
        assert r["prompt_tokens"] + r["max_tokens"] <= 1800  # fits max_model_len 2048
        assert r["ignore_eos"] is True


def test_actual_token_count_matches_target_with_fake_tokenizer():
    for r in build():
        assert r["prompt_tokens_actual"] == r["prompt_tokens"]


def test_file_is_byte_identical(tmp_path):
    h1 = write_jsonl(build(), tmp_path / "a.jsonl")
    h2 = write_jsonl(build(), tmp_path / "b.jsonl")
    assert h1 == h2
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
