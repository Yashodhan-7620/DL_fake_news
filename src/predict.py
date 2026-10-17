"""Inference: load a saved model + its matching preprocessing artefacts (FR-07, FR-08)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .preprocessing import Tokenizer, clean_text
from .utils import LABEL_NAMES, load_config, load_json, resolve

AVAILABLE = ("baseline", "lstm", "gru", "cnn", "bert")
TRAIN_HINT = {"baseline": "python -m src.train_baseline", "lstm": "python -m src.train_neural --model lstm",
              "gru": "python -m src.train_neural --model gru", "cnn": "python -m src.train_neural --model cnn",
              "bert": "python -m src.train_bert"}


class InputError(ValueError):
    """Invalid user input - the message is shown to the user as-is."""


class ModelNotAvailable(FileNotFoundError):
    """Model artefacts are missing - message tells the user how to create them."""


@dataclass
class Prediction:
    label: int
    label_name: str
    score_real: float          # model confidence for class "Real" (NOT a calibrated truth measure)
    model: str


def validate_text(text, max_chars: int, min_chars: int = 20) -> str:
    if text is None or not str(text).strip():
        raise InputError("Please paste some article text first - the input is empty.")
    text = str(text).strip()
    if len(text) < min_chars:
        raise InputError(f"The text is too short ({len(text)} characters). Paste at least {min_chars}.")
    if len(text) > max_chars:
        raise InputError(f"The text is too long ({len(text):,} characters). The limit is {max_chars:,}.")
    return text


def installed_models(cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_config()
    base = resolve(cfg, "models_dir")
    marks = {"baseline": "baseline.joblib", "lstm": "model.pt", "gru": "model.pt",
             "cnn": "model.pt", "bert": "project_config.json"}
    return [m for m in AVAILABLE if (base / m / marks[m]).exists()]


class Predictor:
    def __init__(self, name: str, cfg: dict | None = None):
        if name not in AVAILABLE:
            raise ValueError(f"Unknown model '{name}'. Choose from {AVAILABLE}.")
        self.cfg = cfg or load_config()
        self.name = name
        self.dir = resolve(self.cfg, "models_dir") / name
        self.strip = self.cfg["data"]["strip_source_tags"]
        if name not in installed_models(self.cfg):
            raise ModelNotAvailable(f"Model '{name}' has not been trained yet (no files in {self.dir}). "
                                    f"Run: {TRAIN_HINT[name]}")
        getattr(self, f"_load_{'neural' if name in ('lstm', 'gru', 'cnn') else name}")()

    # ---- loaders -------------------------------------------------------
    def _load_baseline(self):
        import joblib
        bundle = joblib.load(self.dir / "baseline.joblib")
        self.vec, self.clf = bundle["vectorizer"], bundle["model"]

    def _load_neural(self):
        import torch
        from .models import build_model
        self.torch = torch
        self.tok = Tokenizer.load(self.dir / "tokenizer.json")
        conf = load_json(self.dir / "config.json")
        self.net = build_model(self.name, conf["vocab_size"], conf["neural"])
        self.net.load_state_dict(torch.load(self.dir / "model.pt", map_location="cpu"))
        self.net.eval()

    def _load_bert(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.hf_tok = AutoTokenizer.from_pretrained(self.dir)
        self.hf_model = AutoModelForSequenceClassification.from_pretrained(self.dir).eval()
        self.bert_max_len = load_json(self.dir / "project_config.json")["max_len"]

    # ---- inference -----------------------------------------------------
    def _score(self, raw: str) -> float:
        if self.name == "bert":
            enc = self.hf_tok(raw, truncation=True, max_length=self.bert_max_len, return_tensors="pt")
            with self.torch.no_grad():
                logits = self.hf_model(**enc).logits
            return float(self.torch.softmax(logits, dim=-1)[0, 1])
        clean = clean_text(raw, self.strip)
        if not clean:
            raise InputError("After cleaning, no usable words remain in the text. Paste a real article.")
        if self.name == "baseline":
            return float(self.clf.predict_proba(self.vec.transform([clean]))[0, 1])
        x = self.torch.from_numpy(self.tok.encode_batch([clean]))
        with self.torch.no_grad():
            return float(self.torch.sigmoid(self.net(x))[0])

    def predict(self, text: str) -> Prediction:
        text = validate_text(text, self.cfg["data"]["max_chars_input"])
        s = self._score(text)
        label = int(s >= 0.5)
        return Prediction(label, LABEL_NAMES[label], s, self.name)
