"""One command for the whole offline pipeline: audit -> baseline -> LSTM -> CNN (-> GRU) -> report."""
from __future__ import annotations

import argparse

from . import train_baseline, train_neural
from .data_loader import DataError, prepare_data
from .evaluate import comparison_table
from .utils import load_config, resolve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--models", nargs="+", default=["lstm", "cnn", "gru"], choices=["lstm", "cnn", "gru"])
    ap.add_argument("--force-data", action="store_true", help="rebuild cleaned data + splits")
    ap.add_argument("--bert", action="store_true", help="also fine-tune BERT (needs transformers + internet)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    try:
        df = prepare_data(cfg, force=a.force_data)
    except DataError as e:
        print(f"\nDATA ERROR: {e}")
        return 1
    print(f"Data ready: {df['split'].value_counts().to_dict()}  (audit -> reports/data_audit.json)")
    train_baseline.run(cfg)
    for m in a.models:
        train_neural.run(cfg, m)
    if a.bert:
        from . import train_bert
        train_bert.run(cfg)
    table = comparison_table(resolve(cfg, "reports_dir"))
    table.to_csv(resolve(cfg, "reports_dir") / "model_comparison.csv", index=False)
    import glob
    import pandas as pd
    files = glob.glob(str(resolve(cfg, "reports_dir") / "error_analysis_*.csv"))
    if files:
        pd.concat([pd.read_csv(f) for f in files]).to_csv(resolve(cfg, "reports_dir") / "error_analysis.csv", index=False)
    print("\n=== Model comparison (same held-out test split) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
