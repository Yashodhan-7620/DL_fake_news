"""Load, audit, clean, de-duplicate and split the dataset (FR-01 .. FR-03)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessing import clean_text, strip_source_tags
from .utils import LABEL_NAMES, ROOT, resolve, save_json


class DataError(Exception):
    """Raised for missing files / malformed datasets (reliability requirement)."""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataError(
            f"Dataset file not found: {path}\n"
            "Download the ISOT Fake News dataset (Fake.csv, True.csv) into data/raw/, "
            "or run `python scripts/make_sample_data.py` for a synthetic smoke-test dataset."
        )
    try:
        # engine=python + on_bad_lines=skip tolerates malformed rows (reported via row counts)
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception as e:  # pragma: no cover
        raise DataError(f"Could not parse {path}: {e}") from e


def load_raw(cfg: dict) -> pd.DataFrame:
    """Return a frame with columns: title, text, label (0 fake / 1 real)."""
    d = cfg["data"]
    raw_dir = resolve(cfg, "raw_dir")
    t, x = d["title_col"], d["text_col"]
    if d.get("single_csv"):
        df = _read_csv(raw_dir / d["single_csv"])
        for col in (x, d["label_col"]):
            if col not in df.columns:
                raise DataError(f"Column '{col}' not in {d['single_csv']}: {list(df.columns)}")
        df = df.rename(columns={x: "text", d["label_col"]: "label"})
        df["title"] = df[t] if t in df.columns else ""
    else:
        fake = _read_csv(raw_dir / d["fake_file"])
        true = _read_csv(raw_dir / d["true_file"])
        for name, frame in (("fake", fake), ("true", true)):
            if x not in frame.columns:
                raise DataError(f"Column '{x}' missing in {name} file: {list(frame.columns)}")
        fake["label"], true["label"] = 0, 1
        df = pd.concat([fake, true], ignore_index=True)
        df = df.rename(columns={x: "text"})
        df["title"] = df[t] if t in df.columns else ""
    # keep ONLY text fields + label: subject/date/id are potential shortcuts (Sec. 7)
    return df[["title", "text", "label"]].copy()


def audit_and_clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Report data quality problems explicitly and return the cleaned frame + audit dict."""
    a: dict = {"rows_raw": int(len(df)), "columns": list(df.columns)}
    a["label_values_raw"] = {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).items()}
    a["missing_title"] = int(df["title"].isna().sum())
    a["missing_text"] = int(df["text"].isna().sum())
    a["missing_label"] = int(df["label"].isna().sum())

    df = df.dropna(subset=["label"]).copy()
    bad = ~df["label"].isin([0, 1])
    a["unexpected_label_rows_dropped"] = int(bad.sum())
    df = df[~bad].copy()
    df["label"] = df["label"].astype(int)

    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    # audit the publisher-tag shortcut on the RAW text, then (optionally) remove it
    a["source_tag_in_raw_text"] = {
        LABEL_NAMES[k]: int(df.loc[df["label"] == k, "text"].str.contains(r"\(Reuters\)", case=False).sum())
        for k in (0, 1)}
    if cfg["data"]["strip_source_tags"]:
        df["text"] = df["text"].map(strip_source_tags)
    if cfg["data"]["use_title"]:
        df["content"] = (df["title"].str.strip() + ". " + df["text"].str.strip()).str.strip(". ")
    else:
        df["content"] = df["text"].str.strip()

    short = df["content"].str.len() < cfg["data"]["min_chars"]
    a["empty_or_too_short_dropped"] = int(short.sum())
    df = df[~short].copy()

    # conflicting labels: identical content under both labels -> remove every copy
    nlab = df.groupby("content")["label"].nunique()
    conflicts = set(nlab[nlab > 1].index)
    a["conflicting_label_contents"] = len(conflicts)
    df = df[~df["content"].isin(conflicts)].copy()

    # exact duplicates are removed BEFORE splitting (prevents train/test leakage)
    a["exact_duplicates_removed"] = int(df.duplicated(subset="content").sum())
    df = df.drop_duplicates(subset="content").copy()

    # model input text (cleaned); keep the readable original for error analysis
    strip = cfg["data"]["strip_source_tags"]
    df["clean"] = df["content"].map(lambda t: clean_text(t, strip))
    empty_after = df["clean"].str.len() == 0
    a["empty_after_cleaning_dropped"] = int(empty_after.sum())
    df = df[~empty_after].copy()

    # near-duplicates (same first 300 cleaned chars) are reported and removed too
    key = df["clean"].str[:300]
    a["near_duplicates_removed"] = int(key.duplicated().sum())
    df = df[~key.duplicated()].copy()

    a["rows_clean"] = int(len(df))
    a["class_counts"] = {LABEL_NAMES[k]: int(v) for k, v in df["label"].value_counts().sort_index().items()}
    lens = df["clean"].str.split().str.len()
    a["length_words"] = {"mean": float(lens.mean()), "median": float(lens.median()),
                         "p95": float(lens.quantile(0.95)), "max": int(lens.max())}
    if len(a["class_counts"]) < 2:
        raise DataError("Only one class left after cleaning - cannot train a classifier.")

    n = cfg["data"].get("sample_per_class")
    if n:
        df = pd.concat([df[df["label"] == k].sample(min(int((df["label"] == k).sum()), n),
                                                    random_state=cfg["seed"]) for k in (0, 1)])
        a["subsampled_per_class"] = int(n)

    return df.reset_index(drop=True), a


def make_splits(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Stratified train/val/test split with a fixed seed. Adds a 'split' column."""
    s = cfg["data"]["splits"]
    total = s["train"] + s["val"] + s["test"]
    if abs(total - 1.0) > 1e-6:
        raise DataError(f"Split fractions must sum to 1, got {total}")
    seed = cfg["seed"]
    train_idx, rest_idx = train_test_split(
        df.index, test_size=1 - s["train"], stratify=df["label"], random_state=seed)
    val_frac = s["val"] / (s["val"] + s["test"])
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=1 - val_frac, stratify=df.loc[rest_idx, "label"], random_state=seed)
    df = df.copy()
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def prepare_data(cfg: dict, force: bool = False) -> pd.DataFrame:
    """Full data pipeline. Cached to data/processed/dataset.csv so every model sees identical splits."""
    out_dir = resolve(cfg, "processed_dir")
    cache = out_dir / "dataset.csv"
    if cache.exists() and not force:
        df = pd.read_csv(cache, keep_default_na=False)
        df["label"] = df["label"].astype(int)
        return df
    df, audit = audit_and_clean(load_raw(cfg), cfg)
    df = make_splits(df, cfg)
    leak = set(df.loc[df.split == "train", "clean"]) & set(df.loc[df.split == "test", "clean"])
    audit["train_test_exact_overlap"] = len(leak)
    audit["split_sizes"] = {k: int(v) for k, v in df["split"].value_counts().items()}
    audit["split_class_counts"] = {
        sp: {LABEL_NAMES[k]: int(v) for k, v in g["label"].value_counts().sort_index().items()}
        for sp, g in df.groupby("split")}
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    save_json(audit, resolve(cfg, "reports_dir") / "data_audit.json")
    return df


def get_split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    part = df[df["split"] == name]
    if part.empty:
        raise DataError(f"Split '{name}' is empty.")
    return part
