"""Text cleaning, a train-only tokenizer (vocabulary) and padding."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np

from .utils import load_json, save_json

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

_URL = re.compile(r"https?://\S+|www\.\S+")
_TAG = re.compile(r"<[^>]+>")
_NON_WORD = re.compile(r"[^a-z0-9\s']")
_SPACES = re.compile(r"\s+")
# "WASHINGTON (Reuters) -", "(Reuters) -", "21st Century Wire says", "Featured image via ..."
_SOURCE_TAG = re.compile(
    r"^\s*(?:[A-Z][A-Za-z .,/&'-]{0,40}\s)?\((?:Reuters|AP|AFP)\)\s*[-–—]*\s*", re.IGNORECASE
)
_SOURCE_WORDS = re.compile(r"\b(?:reuters|21st century wire|featured image via\b.*)", re.IGNORECASE)


def strip_source_tags(text: str) -> str:
    """Remove publisher fingerprints that would let a model cheat (dataset shortcut)."""
    text = _SOURCE_TAG.sub("", text)
    return _SOURCE_WORDS.sub(" ", text)


def clean_text(text: str, strip_sources: bool = True) -> str:
    """Deterministic normalisation: same input -> same output."""
    if not isinstance(text, str):
        return ""
    if strip_sources:
        text = strip_source_tags(text)
    text = _TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.lower()
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return text.split()


class Tokenizer:
    """Word-level tokenizer. The vocabulary is fitted on TRAINING text only."""

    def __init__(self, vocab_size: int = 30000, min_freq: int = 2, max_len: int = 300):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.max_len = max_len
        self.word2id: dict[str, int] = {PAD: PAD_ID, UNK: UNK_ID}

    # ---- fitting -------------------------------------------------------
    def fit(self, texts) -> "Tokenizer":
        counts = Counter()
        for t in texts:
            counts.update(tokenize(t))
        self.word2id = {PAD: PAD_ID, UNK: UNK_ID}
        for word, c in counts.most_common(self.vocab_size - 2):
            if c < self.min_freq:
                break
            self.word2id[word] = len(self.word2id)
        return self

    @property
    def size(self) -> int:
        return len(self.word2id)

    # ---- encoding ------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        ids = [self.word2id.get(w, UNK_ID) for w in tokenize(text)]
        return ids[: self.max_len]                    # truncate

    def encode_batch(self, texts) -> np.ndarray:
        out = np.full((len(texts), self.max_len), PAD_ID, dtype=np.int64)   # pad
        for i, t in enumerate(texts):
            ids = self.encode(t)
            out[i, : len(ids)] = ids
        return out

    # ---- persistence ---------------------------------------------------
    def save(self, path) -> None:
        save_json(
            {"vocab_size": self.vocab_size, "min_freq": self.min_freq,
             "max_len": self.max_len, "word2id": self.word2id},
            path,
        )

    @classmethod
    def load(cls, path) -> "Tokenizer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer file missing: {path}. Train the model first.")
        d = load_json(path)
        tok = cls(d["vocab_size"], d["min_freq"], d["max_len"])
        tok.word2id = d["word2id"]
        return tok
