import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.utils import load_config  # noqa: E402


@pytest.fixture(scope="session")
def tiny_cfg(tmp_path_factory):
    """Isolated config + synthetic data in a temp dir: tests never touch the real project data."""
    import make_sample_data as msd
    tmp = tmp_path_factory.mktemp("proj")
    raw = tmp / "raw"; raw.mkdir()
    rng = random.Random(1)
    msd.make(150, False, rng).to_csv(raw / "Fake.csv", index=False)
    msd.make(150, True, rng).to_csv(raw / "True.csv", index=False)
    cfg = load_config()
    cfg["data"]["raw_dir"] = str(raw)
    cfg["paths"] = {"models_dir": str(tmp / "models"), "reports_dir": str(tmp / "reports"),
                    "processed_dir": str(tmp / "processed")}
    cfg["neural"].update({"epochs": 2, "max_len": 60, "vocab_size": 3000, "embed_dim": 16,
                          "batch_size": 32, "lstm": {"hidden": 16}, "gru": {"hidden": 16},
                          "cnn": {"filters": 16, "kernel_sizes": [3, 4, 5]}})
    cfg["baseline"].update({"max_features": 2000, "C_grid": [1.0]})
    return cfg
