import numpy as np
import pytest

from src.preprocessing import PAD_ID, UNK_ID, Tokenizer, clean_text, strip_source_tags


def test_clean_is_deterministic_and_normalises():
    t = "Hello,   WORLD!! Visit https://x.com <b>now</b>"
    assert clean_text(t) == clean_text(t) == "hello world visit now"


def test_clean_handles_non_string():
    assert clean_text(None) == "" and clean_text(float("nan")) == ""


def test_source_tag_is_stripped():
    assert strip_source_tags("WASHINGTON (Reuters) - The senator said").strip() == "The senator said"
    assert "reuters" not in clean_text("NEW YORK (Reuters) - Prices rose, Reuters reported")


def test_tokenizer_fits_on_train_only_and_unknown_maps_to_unk():
    tok = Tokenizer(vocab_size=100, min_freq=1, max_len=5).fit(["apple banana apple"])
    assert "apple" in tok.word2id and "cherry" not in tok.word2id
    assert tok.encode("cherry apple")[0] == UNK_ID


def test_min_freq_and_vocab_cap():
    tok = Tokenizer(vocab_size=100, min_freq=2, max_len=5).fit(["a a b"])
    assert "a" in tok.word2id and "b" not in tok.word2id
    small = Tokenizer(vocab_size=3, min_freq=1, max_len=5).fit(["a a b c"])
    assert small.size == 3


def test_padding_and_truncation():
    tok = Tokenizer(100, 1, max_len=4).fit(["a b c d e f"])
    arr = tok.encode_batch(["a b", "a b c d e f"])
    assert arr.shape == (2, 4)
    assert list(arr[0][2:]) == [PAD_ID, PAD_ID] and (arr[1] != PAD_ID).all()


def test_tokenizer_roundtrip(tmp_path):
    tok = Tokenizer(100, 1, 6).fit(["x y z"])
    tok.save(tmp_path / "t.json")
    tok2 = Tokenizer.load(tmp_path / "t.json")
    assert tok2.word2id == tok.word2id and np.array_equal(tok.encode_batch(["x z"]), tok2.encode_batch(["x z"]))


def test_missing_tokenizer_file_message(tmp_path):
    with pytest.raises(FileNotFoundError):
        Tokenizer.load(tmp_path / "nope.json")
