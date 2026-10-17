import pandas as pd
import pytest

from src.data_loader import DataError, audit_and_clean, load_raw, make_splits, prepare_data
from src.evaluate import compute_metrics


def _cfg(tiny_cfg, **over):
    cfg = {**tiny_cfg, "data": {**tiny_cfg["data"], **over}}
    return cfg


def test_missing_file_gives_helpful_error(tiny_cfg, tmp_path):
    cfg = _cfg(tiny_cfg, raw_dir=str(tmp_path))
    with pytest.raises(DataError, match="not found"):
        load_raw(cfg)


def test_audit_handles_missing_dupes_conflicts_and_bad_labels(tiny_cfg):
    long = "this is a sufficiently long article body about policy and trade. "
    df = pd.DataFrame({
        "title": ["a", "a", "c", "c", None, "f", "g"],
        "text": [long + "one", long + "one", long + "two", long + "two", long + "three", "", long + "four"],
        "label": [0, 0, 0, 1, 1, 1, 7],       # row 2 duplicate, rows 3/4 conflict, row 6 empty, row 7 bad label
    })
    out, a = audit_and_clean(df, tiny_cfg)
    assert a["unexpected_label_rows_dropped"] == 1
    assert a["empty_or_too_short_dropped"] >= 1
    assert a["conflicting_label_contents"] == 1
    assert a["exact_duplicates_removed"] == 1
    assert out["clean"].is_unique


def test_split_is_stratified_disjoint_and_sums(tiny_cfg):
    df = prepare_data(tiny_cfg, force=True)
    assert set(df["split"]) == {"train", "val", "test"}
    tr, te = set(df[df.split == "train"]["clean"]), set(df[df.split == "test"]["clean"])
    assert not (tr & te)                                              # no leakage
    ratios = df.groupby("split")["label"].mean()
    assert ratios.max() - ratios.min() < 0.1                          # stratified


def test_split_deterministic(tiny_cfg):
    base = prepare_data(tiny_cfg, force=True)
    again = make_splits(base.drop(columns="split"), tiny_cfg)
    assert (base["split"].values == again["split"].values).all()


def test_bad_split_fractions(tiny_cfg):
    cfg = _cfg(tiny_cfg, splits={"train": 0.5, "val": 0.2, "test": 0.2})
    with pytest.raises(DataError):
        make_splits(prepare_data(tiny_cfg).drop(columns="split"), cfg)


def test_metrics_handle_both_classes_and_order():
    m = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], [0.1, 0.6, 0.9, 0.8])
    assert m["confusion_matrix"] == [[1, 1], [0, 2]]
    assert m["confusion_matrix_labels"] == ["Fake", "Real"]
    assert 0 < m["macro_f1"] <= 1 and m["roc_auc"] == 1.0


def test_metrics_single_class_does_not_crash():
    m = compute_metrics([1, 1], [1, 1], [0.9, 0.8])
    assert m["roc_auc"] is None and m["accuracy"] == 1.0
