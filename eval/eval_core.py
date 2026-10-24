"""Pure helpers for eval/run_eval.py.

No torch and no lm_eval imported here, so everything in this file can be
unit-tested on a laptop with no GPU and no internet.
"""
import hashlib
import json
from pathlib import Path

FILTER = "none"  # lm-eval names metrics "<metric>,<filter>"; these tasks use "none"


def json_default(obj):
    """Fallback for json.dump: numpy / torch values and anything else odd."""
    if hasattr(obj, "tolist"):       # numpy arrays AND numpy scalars, torch tensors
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def build_model_args(cfg):
    """lm-eval model_args string. Direct indexing: a missing key fails loudly."""
    ev = cfg["evaluation"]
    return f"pretrained={cfg['model']},dtype={ev['dtype']},max_length={ev['max_length']}"


def seed_kwargs(cfg):
    """Map the config's seeds onto simple_evaluate's four seed parameters."""
    s = cfg["evaluation"]["seeds"]
    return {
        "random_seed": s["python"],
        "numpy_random_seed": s["numpy"],
        "torch_random_seed": s["torch"],
        "fewshot_random_seed": s["fewshot"],
    }


def _as_float(value, what):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} is not a number: {value!r}")


def extract_metrics(raw, ev):
    """Pull the configured metrics out of lm-eval's results dict.

    Returns {task: {metric: value, metric_stderr: value}}.
    Fails loudly if a configured task or metric is missing. A stderr that
    lm-eval reports as "N/A" (wikitext perplexity) is skipped, not invented.
    """
    results = raw["results"]
    out = {}
    for task in ev["tasks"]:
        if task not in results:
            raise KeyError(f"task '{task}' missing from results (have: {sorted(results)})")
        got = results[task]
        spec = ev["metrics"][task]
        if spec["headline"] not in spec["report"]:
            raise ValueError(f"headline '{spec['headline']}' for '{task}' is not in its report list")
        out[task] = {}
        for metric in spec["report"]:
            key = f"{metric},{FILTER}"
            if key not in got:
                raise KeyError(f"'{key}' missing for task '{task}' (have: {sorted(got)})")
            out[task][metric] = _as_float(got[key], f"{task}/{key}")
            if ev["report_stderr"]:
                skey = f"{metric}_stderr,{FILTER}"
                if skey not in got:
                    raise KeyError(f"'{skey}' missing for task '{task}' (have: {sorted(got)})")
                if got[skey] == "N/A":
                    continue
                out[task][f"{metric}_stderr"] = _as_float(got[skey], f"{task}/{skey}")
    return out


def headline_summary(extracted, ev):
    """{task: (headline_metric, value, stderr or None)} for printing."""
    summary = {}
    for task in ev["tasks"]:
        m = ev["metrics"][task]["headline"]
        summary[task] = (m, extracted[task][m], extracted[task].get(f"{m}_stderr"))
    return summary


def write_samples(path, samples_by_task, ev):
    """One compact json line per logged sample. Deterministic bytes.

    Returns (number_of_records, records_without_metrics, sha256).
    """
    lines, without = [], 0
    for task in ev["tasks"]:
        if task not in samples_by_task:
            raise KeyError(f"no logged samples for task '{task}' (is log_samples on?)")
        report = ev["metrics"][task]["report"]
        for s in samples_by_task[task]:
            metrics = {m: s[m] for m in report if m in s}
            if not metrics:
                without += 1
            rec = {"task": task, "doc_id": s.get("doc_id"),
                   "filter": s.get("filter"), "metrics": metrics}
            lines.append(json.dumps(rec, sort_keys=True, default=json_default))
    data = ("\n".join(lines) + "\n" if lines else "").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(lines), without, hashlib.sha256(data).hexdigest()
