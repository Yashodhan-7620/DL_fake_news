"""End-to-end checks on a tiny synthetic dataset: train -> save -> load -> predict."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src import train_baseline, train_neural
from src.models import build_model
from src.predict import InputError, ModelNotAvailable, Predictor, installed_models, validate_text
from src.utils import load_json


def test_model_output_shapes(tiny_cfg):
    x = torch.randint(0, 50, (4, 60))
    x[0, 30:] = 0                                              # padded row
    x[1] = 0                                                   # all-pad row must not crash
    for name in ("lstm", "gru", "cnn"):
        net = build_model(name, 50, tiny_cfg["neural"])
        assert net(x).shape == (4,)


def test_baseline_trains_and_saves(tiny_cfg):
    m = train_baseline.run(tiny_cfg)
    assert 0 <= m["accuracy"] <= 1
    assert "baseline" in installed_models(tiny_cfg)


@pytest.mark.parametrize("name", ["lstm", "gru", "cnn"])
def test_neural_trains_saves_loads_and_is_deterministic(tiny_cfg, name):
    m = train_neural.run(tiny_cfg, name)
    assert m["confusion_matrix_labels"] == ["Fake", "Real"]
    assert name in installed_models(tiny_cfg)
    p1 = Predictor(name, tiny_cfg).predict("The finance minister said the budget bill passed by a narrow margin.")
    p2 = Predictor(name, tiny_cfg).predict("The finance minister said the budget bill passed by a narrow margin.")
    assert p1.score_real == pytest.approx(p2.score_real) and 0 <= p1.score_real <= 1


def test_all_models_share_the_same_test_split(tiny_cfg):
    metrics = load_json(f"{tiny_cfg['paths']['reports_dir']}/metrics.json")
    supports = {tuple(v["per_class"][c]["support"] for c in ("Fake", "Real")) for v in metrics.values()}
    assert len(supports) == 1


def test_loss_decreases_during_training(tiny_cfg):
    hist = load_json(f"{tiny_cfg['paths']['models_dir']}/cnn/history.json")
    assert hist["train_loss"][-1] <= hist["train_loss"][0]


def test_baseline_predict_and_input_validation(tiny_cfg):
    p = Predictor("baseline", tiny_cfg)
    assert p.predict("Officials said the ministry would publish a detailed report next quarter.").label in (0, 1)
    for bad in ("", "   \n\t", None, "short"):
        with pytest.raises(InputError):
            p.predict(bad)
    with pytest.raises(InputError, match="too long"):
        validate_text("x" * 50000, tiny_cfg["data"]["max_chars_input"])
    with pytest.raises(InputError, match="no usable words"):
        p.predict("!!! ??? ### $$$ %%% &&& *** !!! ??? ###")


def test_missing_model_artefacts_message(tiny_cfg, tmp_path):
    empty = {**tiny_cfg, "paths": {**tiny_cfg["paths"], "models_dir": str(tmp_path / "none")}}
    for name in ("baseline", "lstm", "bert"):
        with pytest.raises(ModelNotAvailable, match="has not been trained"):
            Predictor(name, empty)


def test_unknown_model_name(tiny_cfg):
    with pytest.raises(ValueError):
        Predictor("nonsense", tiny_cfg)
