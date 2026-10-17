"""OPTIONAL: fine-tune DistilBERT/BERT (needs `transformers` + internet for the pretrained weights).

`--smoke` builds a tiny randomly-initialised BERT with a local vocabulary instead, so the code path
can be verified offline. Smoke results are meaningless - it only proves the pipeline runs.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .data_loader import get_split, prepare_data
from .evaluate import compute_metrics, error_table, plot_confusion, update_metrics_file
from .utils import environment_info, load_config, resolve, save_json, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _smoke_model(train_texts, out_dir, max_len):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    words = sorted({w for t in train_texts for w in t.split()})[:2000]
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = out_dir / "vocab.txt"
    vf.write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(str(vf), do_lower_case=True)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=max(64, max_len), num_labels=2)
    return tok, BertForSequenceClassification(cfg)


def run(cfg: dict, smoke: bool = False) -> dict:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        raise SystemExit("The BERT extension needs `pip install transformers`.") from e
    set_seed(cfg["seed"])
    b = cfg["bert"]
    df = prepare_data(cfg)
    train, val, test = (get_split(df, s) for s in ("train", "val", "test"))
    out = resolve(cfg, "models_dir") / "bert"
    rep = resolve(cfg, "reports_dir")
    max_len = 64 if smoke else b["max_len"]
    # BERT uses its OWN tokenizer; feed lightly-cleaned text (keep case/punctuation for wordpiece)
    col = "clean" if smoke else "content"
    if smoke:
        tok, model = _smoke_model(train["clean"].tolist(), out, max_len)
    else:
        tok = AutoTokenizer.from_pretrained(b["model_name"])
        model = AutoModelForSequenceClassification.from_pretrained(b["model_name"], num_labels=2)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=b["lr"])
    rng = np.random.default_rng(cfg["seed"])

    def batches(frame, shuffle):
        idx = rng.permutation(len(frame)) if shuffle else np.arange(len(frame))
        for i in range(0, len(idx), b["batch_size"]):
            part = frame.iloc[idx[i:i + b["batch_size"]]]
            enc = tok(part[col].tolist(), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
            yield {k: v.to(DEVICE) for k, v in enc.items()}, torch.tensor(part["label"].to_numpy()).to(DEVICE)

    @torch.no_grad()
    def evaluate(frame):
        model.eval(); losses, probs = [], []
        for enc, y in batches(frame, False):
            lg = model(**enc).logits
            losses.append(torch.nn.functional.cross_entropy(lg, y, reduction="sum").item())
            probs.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
        return sum(losses) / len(frame), np.concatenate(probs)

    t0, best, best_state = time.time(), float("inf"), None
    for ep in range(1, (1 if smoke else b["epochs"]) + 1):
        model.train(); tl = 0.0
        for enc, y in batches(train, True):
            loss = torch.nn.functional.cross_entropy(model(**enc).logits, y)
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item() * len(y)
        vl, _ = evaluate(val)
        print(f"  epoch {ep} train_loss={tl / len(train):.4f} val_loss={vl:.4f}")
        if vl < best:
            best, best_state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    _, score = evaluate(test)                               # test touched once, at the end
    pred = (score >= 0.5).astype(int)
    m = compute_metrics(test["label"], pred, score)

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    save_json({"model": "bert", "base": "smoke-tiny-bert" if smoke else b["model_name"], "max_len": max_len,
               "smoke": smoke, "seed": cfg["seed"], "env": environment_info()}, out / "project_config.json")
    name = "bert_smoke" if smoke else "bert"
    plot_confusion(m["confusion_matrix"], rep / "figures" / f"cm_{name}.png", name)
    errs = error_table(test, pred, score, name)
    errs.to_csv(rep / f"error_analysis_{name}.csv", index=False)
    update_metrics_file(rep, name, m, {"train_seconds": round(time.time() - t0, 1), "n_errors": int(len(errs))})
    print(f"[{name}] test acc={m['accuracy']:.4f} macro-F1={m['macro_f1']:.4f}")
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny offline model, pipeline check only")
    a = ap.parse_args()
    run(load_config(a.config), a.smoke)
