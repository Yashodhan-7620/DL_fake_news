"""Train BiLSTM / GRU / CNN with early stopping on VALIDATION loss (FR-05, FR-06)."""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn

from .data_loader import get_split, prepare_data
from .evaluate import (compute_metrics, error_table, plot_confusion, plot_curves, update_metrics_file)
from .models import build_model
from .preprocessing import Tokenizer
from .utils import environment_info, load_config, resolve, save_json, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loader(X, y, batch_size, shuffle, seed):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)


@torch.no_grad()
def predict_proba(model, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(DEVICE)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def _epoch(model, loader, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    tot, n, correct = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if train:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
            correct += ((logits > 0).float() == yb).sum().item()
    return tot / n, correct / n


def run(cfg: dict, name: str) -> dict:
    set_seed(cfg["seed"])
    n = cfg["neural"]
    if n.get("num_threads"):
        torch.set_num_threads(int(n["num_threads"]))
    df = prepare_data(cfg)
    train, val, test = (get_split(df, s) for s in ("train", "val", "test"))

    # vocabulary from TRAIN only
    tok = Tokenizer(n["vocab_size"], n["min_freq"], n["max_len"]).fit(train["clean"])
    Xtr, Xva, Xte = (tok.encode_batch(p["clean"].tolist()) for p in (train, val, test))
    ytr, yva, yte = (p["label"].to_numpy() for p in (train, val, test))

    model = build_model(name, tok.size, n).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] device={DEVICE} vocab={tok.size} params={n_params:,} "
          f"train/val/test={len(train)}/{len(val)}/{len(test)}")
    opt = torch.optim.Adam(model.parameters(), lr=n["lr"])
    loss_fn = nn.BCEWithLogitsLoss()
    tr_loader = _loader(Xtr, ytr, n["batch_size"], True, cfg["seed"])
    va_loader = _loader(Xva, yva, n["batch_size"], False, cfg["seed"])

    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_state, best_val, bad, t0 = None, float("inf"), 0, time.time()
    for ep in range(1, n["epochs"] + 1):
        tl, _ = _epoch(model, tr_loader, loss_fn, opt)
        vl, va = _epoch(model, va_loader, loss_fn)
        hist["train_loss"].append(tl); hist["val_loss"].append(vl); hist["val_acc"].append(va)
        print(f"  epoch {ep:02d} train_loss={tl:.4f} val_loss={vl:.4f} val_acc={va:.4f}")
        if vl < best_val - 1e-4:
            best_val, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= n["patience"]:
                print("  early stopping (validation loss stopped improving)")
                break
    model.load_state_dict(best_state)

    # the TEST set is touched exactly once, after model selection is finished
    score = predict_proba(model, Xte)
    pred = (score >= 0.5).astype(int)
    m = compute_metrics(yte, pred, score)

    out = resolve(cfg, "models_dir") / name
    rep = resolve(cfg, "reports_dir")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    tok.save(out / "tokenizer.json")
    save_json({"model": name, "vocab_size": tok.size, "neural": n, "seed": cfg["seed"],
               "best_val_loss": best_val, "epochs_run": len(hist["train_loss"]),
               "env": environment_info()}, out / "config.json")
    save_json(hist, out / "history.json")
    plot_confusion(m["confusion_matrix"], rep / "figures" / f"cm_{name}.png", f"{name.upper()}")
    plot_curves(hist, rep / "figures" / f"curves_{name}.png", name.upper())
    errs = error_table(test, pred, score, name)
    errs.to_csv(rep / f"error_analysis_{name}.csv", index=False)
    update_metrics_file(rep, name, m, {"train_seconds": round(time.time() - t0, 1), "params": n_params,
                                       "best_val_loss": best_val, "n_errors": int(len(errs))})
    print(f"[{name}] test acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["lstm", "gru", "cnn"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    run(load_config(a.config), a.model)
