import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src import train_bert
from src.predict import Predictor


def test_bert_smoke_trains_saves_and_predicts(tiny_cfg):
    cfg = {**tiny_cfg, "bert": {**tiny_cfg["bert"], "batch_size": 32}}
    train_bert.run(cfg, smoke=True)
    r = Predictor("bert", cfg).predict("Officials said the ministry would publish a detailed report soon.")
    assert 0 <= r.score_real <= 1
