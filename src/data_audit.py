"""Data audit report + figures (FR-01, FR-02).  Run: python -m src.data_audit"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_loader import DataError, prepare_data
from .utils import LABEL_NAMES, load_config, resolve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    try:
        df = prepare_data(cfg, force=a.force)
    except DataError as e:
        print(f"DATA ERROR: {e}")
        return 1
    rep = resolve(cfg, "reports_dir")
    audit = json.load(open(rep / "data_audit.json")) if (rep / "data_audit.json").exists() else {}
    print(json.dumps(audit, indent=2))
    fig_dir = rep / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    counts = df["label"].map(LABEL_NAMES).value_counts()
    ax[0].bar(counts.index, counts.values, color=["#c0392b", "#2980b9"][: len(counts)]); ax[0].set_title("Class balance")
    for k in (0, 1):
        ax[1].hist(df.loc[df.label == k, "clean"].str.split().str.len().clip(upper=2000), bins=40,
                   alpha=0.6, label=LABEL_NAMES[k])
    ax[1].set_title("Article length (words)"); ax[1].legend()
    fig.tight_layout(); fig.savefig(fig_dir / "data_audit.png", dpi=140)
    print(f"\nSaved reports/data_audit.json and figures/data_audit.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
