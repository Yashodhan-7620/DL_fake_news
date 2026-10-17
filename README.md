# Fake News Detection Using Deep Learning and NLP

Educational text-classification project: TF-IDF + Logistic Regression baseline, BiLSTM, GRU, Text-CNN
(PyTorch) and an optional BERT/DistilBERT extension, plus a Streamlit demo.
**The model predicts dataset labels (Real-like / Fake-like). It does not verify facts.**

## 1. Create the conda environment (not venv)
```bash
conda env create -f environment.yml      # creates env "fakenews" (Python 3.11)
conda activate fakenews
python -c "import torch, sklearn, streamlit; print(torch.__version__)"
```
NVIDIA GPU (optional): after creating the env, replace PyTorch with a CUDA build, e.g.
`conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia` (pick the version from pytorch.org).
Update later: `conda env update -f environment.yml --prune`. Remove: `conda env remove -n fakenews`.

## 2. Get the dataset
Download the **ISOT Fake News Dataset** (University of Victoria; check its terms) and place
`Fake.csv` and `True.csv` in `data/raw/`. Record the source URL, access date and licence in your report.
No internet / just testing the code? `python scripts/make_sample_data.py` writes a SYNTHETIC dataset
(scores on it are meaningless, ~100%). Delete it before using real data, then run with `--force-data`.

## 3. Run the pipeline
```bash
python -m src.data_audit                 # data audit -> reports/data_audit.json + figure
python -m src.run_all                    # audit + baseline + LSTM + CNN + GRU + comparison
python -m src.train_baseline             # or train models individually:
python -m src.train_neural --model lstm  #   lstm | gru | cnn
python -m src.train_bert                 # optional, needs internet for pretrained weights
streamlit run app.py                     # demo at http://localhost:8501
pytest -q                                # 29 tests
```
CPU-friendly quick run: set `data.sample_per_class: 3000` and `neural.epochs: 4` in `configs/config.yaml`,
then rerun with `--force-data`. After changing data settings always use `--force-data`.

## 4. What the pipeline does (and why)
| Stage | Implementation | Requirement |
|---|---|---|
| Load + validate | `data_loader.py` (clear errors for missing files/columns/bad rows) | FR-01 |
| Clean | missing/short rows, conflicting labels, exact + near duplicates removed **before** splitting | FR-02 |
| Split | stratified 70/15/15, fixed seed, cached in `data/processed/dataset.csv` | FR-03 |
| Leakage control | tokenizer / TF-IDF fitted on train only; test used once at the end; publisher tags like `(Reuters)` stripped (ISOT shortcut) | FR-03, Sec. 7 |
| Baseline | TF-IDF (1-2 grams) + LR, C tuned on validation only | FR-04 |
| Neural | BiLSTM, GRU, TextCNN; BCE loss; early stopping on val loss | FR-05 |
| Compare | `reports/metrics.json`, `model_comparison.csv`, confusion matrices, curves | FR-06, FR-11 |
| App | input validation, disclaimer, model selector, graceful missing-model errors | FR-07..09 |
| Reproducibility | seed + config + library versions saved next to every model | FR-10 |

Artefacts: `models/<name>/` (weights, tokenizer, config), `reports/` (metrics, figures, error CSVs).
Label convention everywhere: `0 = Fake`, `1 = Real`; the score shown is P(Real).

## 5. Limitations
Dataset-defined labels, possible source/topic shortcuts, no claim of truth, performance may drop on other
publishers or time periods. High scores on ISOT should be read with this in mind. Pasted text is processed
locally and never stored.
