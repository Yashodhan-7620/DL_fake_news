"""Metrics, confusion matrices, learning curves and error analysis (FR-11)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # headless-safe (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)

from .utils import LABEL_NAMES

ORDER = [0, 1]                              # fixed label order everywhere


def compute_metrics(y_true, y_pred, y_score=None) -> dict:
    """y_score = probability of class 1 (Real). Handles degenerate single-class cases."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=ORDER, zero_division=0)
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean(f)),
        "per_class": {LABEL_NAMES[k]: {"precision": float(p[i]), "recall": float(r[i]),
                                       "f1": float(f[i]), "support": int(s[i])}
                      for i, k in enumerate(ORDER)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ORDER).tolist(),
        "confusion_matrix_labels": [LABEL_NAMES[k] for k in ORDER],
    }
    if y_score is not None and len(set(y_true.tolist())) == 2:
        m["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        m["roc_auc"] = None
    return m


def plot_confusion(cm, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3.4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=[LABEL_NAMES[k] for k in ORDER], yticklabels=[LABEL_NAMES[k] for k in ORDER])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_curves(history: dict, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    ep = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(ep, history["train_loss"], label="train"); axes[0].plot(ep, history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(ep, history["val_acc"], label="val acc"); axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("epoch")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def error_table(df_test: pd.DataFrame, y_pred, y_score, model: str) -> pd.DataFrame:
    """Misclassified test rows for error analysis (redact/inspect manually)."""
    out = df_test[["title", "text", "label"]].copy()
    out["pred"] = np.asarray(y_pred)
    out["score_real"] = np.asarray(y_score)
    out["model"] = model
    out = out[out["label"] != out["pred"]].copy()
    out["actual_name"] = out["label"].map(LABEL_NAMES)
    out["pred_name"] = out["pred"].map(LABEL_NAMES)
    out["text"] = out["text"].str.slice(0, 400)
    return out


def update_metrics_file(reports_dir: Path, model: str, metrics: dict, extra: dict | None = None) -> None:
    """Merge one model's metrics into reports/metrics.json (all models, same test split)."""
    from .utils import load_json, save_json
    path = reports_dir / "metrics.json"
    allm = load_json(path) if path.exists() else {}
    allm[model] = {**metrics, **(extra or {})}
    save_json(allm, path)


def comparison_table(reports_dir: Path) -> pd.DataFrame:
    from .utils import load_json
    path = reports_dir / "metrics.json"
    if not path.exists():
        raise FileNotFoundError("reports/metrics.json not found - train at least one model first.")
    rows = []
    for name, m in load_json(path).items():
        rows.append({"model": name, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "roc_auc": m.get("roc_auc"),
                     "fake_recall": m["per_class"]["Fake"]["recall"],
                     "real_recall": m["per_class"]["Real"]["recall"],
                     "train_seconds": m.get("train_seconds")})
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
