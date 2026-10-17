"""TF-IDF + Logistic Regression baseline (FR-04)."""
from __future__ import annotations

import argparse
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from .data_loader import get_split, prepare_data
from .evaluate import (compute_metrics, error_table, plot_confusion, update_metrics_file)
from .utils import environment_info, load_config, resolve, save_json, set_seed


def run(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    df = prepare_data(cfg)
    train, val, test = (get_split(df, s) for s in ("train", "val", "test"))
    b = cfg["baseline"]
    t0 = time.time()

    # vectoriser is fitted on TRAIN only (FR-03)
    vec = TfidfVectorizer(max_features=b["max_features"], ngram_range=tuple(b["ngram_range"]),
                          sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform(train["clean"])
    Xva, Xte = vec.transform(val["clean"]), vec.transform(test["clean"])

    # C is tuned on the VALIDATION set only
    best_c, best_f1, best_clf = None, -1.0, None
    for c in b["C_grid"]:
        clf = LogisticRegression(C=c, max_iter=1000, random_state=cfg["seed"])
        clf.fit(Xtr, train["label"])
        f1 = f1_score(val["label"], clf.predict(Xva), average="macro")
        print(f"  C={c:<6} val macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_c, best_f1, best_clf = c, f1, clf

    pred = best_clf.predict(Xte)
    score = best_clf.predict_proba(Xte)[:, 1]
    m = compute_metrics(test["label"], pred, score)
    rep = resolve(cfg, "reports_dir")
    out = resolve(cfg, "models_dir") / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vec, "model": best_clf}, out / "baseline.joblib")
    config = {"best_C": best_c, "val_macro_f1": best_f1, "max_features": b["max_features"],
              "ngram_range": b["ngram_range"], "seed": cfg["seed"], "env": environment_info()}
    save_json(config, out / "config.json")
    plot_confusion(m["confusion_matrix"], rep / "figures" / "cm_baseline.png", "Baseline (TF-IDF + LR)")
    errs = error_table(test, pred, score, "baseline")
    errs.to_csv(rep / "error_analysis_baseline.csv", index=False)
    update_metrics_file(rep, "baseline", m, {"train_seconds": round(time.time() - t0, 1),
                                             "best_C": best_c, "val_macro_f1": best_f1,
                                             "n_errors": int(len(errs))})
    print(f"[baseline] test acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
