"""Tiny YAML config loader with `extends:` support.

    cfg = load_config("configs/int4.yaml")   # base.yaml merged under int4.yaml
    cfg_hash(cfg)                            # store both in the results json
"""
import copy
import hashlib
import json
from pathlib import Path

import yaml


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    return cfg


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
