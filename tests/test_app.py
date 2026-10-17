"""UI behaviour via Streamlit's headless test runner (no browser needed)."""
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("torch")
from streamlit.testing.v1 import AppTest

from src import train_baseline
from src.utils import ROOT


@pytest.fixture()
def app(tiny_cfg, monkeypatch):
    train_baseline.run(tiny_cfg)
    monkeypatch.setattr("src.utils.load_config", lambda path=None: tiny_cfg)
    monkeypatch.setattr("src.predict.load_config", lambda path=None: tiny_cfg)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    return at.run()


def test_disclaimer_visible_and_no_exception(app):
    assert not app.exception
    assert any("not independent fact" in w.value for w in app.warning)


def test_empty_input_shows_helpful_message(app):
    app.button[0].click().run()
    assert any("empty" in e.value for e in app.error)


def test_valid_input_returns_prediction(app):
    app.text_area[0].set_value("The finance minister said the budget bill passed by a narrow margin today.").run()
    app.button[0].click().run()
    assert not app.exception
    assert any("Predicted dataset label" in s.value for s in app.subheader)
