import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from config import cfg_hash, load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_extends_merges_and_overrides(tmp_path):
    (tmp_path / "base.yaml").write_text("model: a\nserver:\n  dtype: half\n  port: 1\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nmodel: b\nserver:\n  port: 2\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg == {"model": "b", "server": {"dtype": "half", "port": 2}}


def test_hash_stable_and_sensitive():
    c = {"a": 1, "b": {"c": 2}}
    assert cfg_hash(c) == cfg_hash({"b": {"c": 2}, "a": 1})
    assert cfg_hash(c) != cfg_hash({"a": 1, "b": {"c": 3}})


def test_real_variant_configs_differ_only_in_model_and_variant():
    fp16 = load_config(ROOT / "configs/fp16.yaml")
    for v in ("int8", "int4"):
        other = load_config(ROOT / f"configs/{v}.yaml")
        diff = {k for k in fp16 if fp16[k] != other[k]}
        assert diff == {"model", "variant"}
