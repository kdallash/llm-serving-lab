"""Run with: pytest tests/test_eval_core.py   (no GPU, no internet, no lm_eval needed)"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.eval_core import (build_model_args, extract_metrics, headline_summary,  # noqa: E402
                            json_default, seed_kwargs, write_samples)

EV = {
    "tasks": ["wikitext", "arc_easy"],
    "report_stderr": True,
    "metrics": {
        "wikitext": {"headline": "word_perplexity", "report": ["word_perplexity"]},
        "arc_easy": {"headline": "acc_norm", "report": ["acc_norm", "acc"]},
    },
}
RAW = {"results": {
    "wikitext": {"alias": "wikitext", "word_perplexity,none": 20.5,
                 "word_perplexity_stderr,none": "N/A", "byte_perplexity,none": 1.7},
    "arc_easy": {"alias": "arc_easy", "acc,none": 0.6, "acc_stderr,none": 0.01,
                 "acc_norm,none": 0.55, "acc_norm_stderr,none": 0.011},
}}


def test_extract_happy_path():
    out = extract_metrics(RAW, EV)
    assert out["wikitext"] == {"word_perplexity": 20.5}          # "N/A" stderr skipped
    assert out["arc_easy"] == {"acc_norm": 0.55, "acc_norm_stderr": 0.011,
                               "acc": 0.6, "acc_stderr": 0.01}


def test_acc_and_acc_norm_are_not_mixed_up():
    out = extract_metrics(RAW, EV)["arc_easy"]
    assert out["acc"] == 0.6 and out["acc_norm"] == 0.55


def test_missing_task_raises():
    raw = {"results": {"wikitext": RAW["results"]["wikitext"]}}
    with pytest.raises(KeyError):
        extract_metrics(raw, EV)


def test_missing_metric_raises():
    raw = {"results": {**RAW["results"], "arc_easy": {"acc,none": 0.6, "acc_stderr,none": 0.01}}}
    with pytest.raises(KeyError):
        extract_metrics(raw, EV)


def test_missing_stderr_key_raises():
    arc = dict(RAW["results"]["arc_easy"])
    del arc["acc_norm_stderr,none"]
    with pytest.raises(KeyError):
        extract_metrics({"results": {**RAW["results"], "arc_easy": arc}}, EV)


def test_non_numeric_metric_raises():
    arc = dict(RAW["results"]["arc_easy"], **{"acc_norm,none": "N/A"})
    with pytest.raises(ValueError):
        extract_metrics({"results": {**RAW["results"], "arc_easy": arc}}, EV)


def test_headline_must_be_reported():
    ev = {**EV, "metrics": {**EV["metrics"],
                            "arc_easy": {"headline": "acc_norm", "report": ["acc"]}}}
    with pytest.raises(ValueError):
        extract_metrics(RAW, ev)


def test_headline_summary():
    s = headline_summary(extract_metrics(RAW, EV), EV)
    assert s["arc_easy"] == ("acc_norm", 0.55, 0.011)
    assert s["wikitext"] == ("word_perplexity", 20.5, None)


SAMPLES = {
    "wikitext": [{"doc_id": 0, "filter": "none", "word_perplexity": [-100.0, 50]}],
    "arc_easy": [{"doc_id": i, "filter": "none", "acc": float(i % 2), "acc_norm": 1.0}
                 for i in range(5)],
}


def test_write_samples_deterministic(tmp_path):
    a = write_samples(tmp_path / "a.jsonl", SAMPLES, EV)
    b = write_samples(tmp_path / "b.jsonl", SAMPLES, EV)
    assert a == b and a[0] == 6 and a[1] == 0
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


def test_write_samples_counts_records_without_metrics(tmp_path):
    s = {**SAMPLES, "arc_easy": [{"doc_id": 0, "filter": "none"}]}
    assert write_samples(tmp_path / "a.jsonl", s, EV)[1] == 1


def test_write_samples_missing_task_raises(tmp_path):
    with pytest.raises(KeyError):
        write_samples(tmp_path / "a.jsonl", {"wikitext": SAMPLES["wikitext"]}, EV)


def test_json_default_handles_array_like_and_scalar_like():
    class Arr:
        def tolist(self):
            return [1, 2]

    class Scalar:
        def item(self):
            return 3.5

    assert json_default(Arr()) == [1, 2]
    assert json_default(Scalar()) == 3.5
    assert isinstance(json_default(object()), str)


CFG = {"model": "Qwen/Qwen3-0.6B",
       "evaluation": {"dtype": "float16", "max_length": 2048,
                      "seeds": {"python": 1, "numpy": 2, "torch": 3, "fewshot": 4}}}


def test_model_args_come_from_config():
    assert build_model_args(CFG) == "pretrained=Qwen/Qwen3-0.6B,dtype=float16,max_length=2048"


def test_missing_config_key_fails_loudly():
    bad = {"model": "x", "evaluation": {"dtype": "float16"}}
    with pytest.raises(KeyError):
        build_model_args(bad)


def test_seed_mapping():
    assert seed_kwargs(CFG) == {"random_seed": 1, "numpy_random_seed": 2,
                                "torch_random_seed": 3, "fewshot_random_seed": 4}
