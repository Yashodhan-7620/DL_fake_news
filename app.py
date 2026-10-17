"""Streamlit demo.  Run:  streamlit run app.py"""
import streamlit as st

from src.predict import InputError, ModelNotAvailable, Predictor, installed_models, validate_text
from src.utils import load_config

st.set_page_config(page_title="Fake News Detection (educational)", page_icon="📰", layout="centered")

DISCLAIMER = ("This is a machine-learning estimate based on training data, not independent fact "
              "verification. Verify important claims with trusted sources.")
NICE = {"baseline": "TF-IDF + Logistic Regression", "lstm": "BiLSTM", "gru": "GRU", "cnn": "Text CNN",
        "bert": "BERT (fine-tuned)"}


@st.cache_resource(show_spinner=False)
def get_predictor(name: str) -> Predictor:
    return Predictor(name)


cfg = load_config()
st.title("📰 Fake News Detection")
st.caption("Deep Learning & NLP course project. The model predicts whether text *resembles* articles "
           "labelled Real or Fake in its training dataset.")
st.warning(DISCLAIMER)

models = installed_models(cfg)
if not models:
    st.error("No trained models found. Train first, e.g. `python -m src.run_all`, then reload this page.")
    st.stop()

choice = st.selectbox("Model", models, format_func=lambda m: NICE.get(m, m)) if len(models) > 1 else models[0]
text = st.text_area("Article title and body", height=260, placeholder="Paste the article text here...",
                    key="article")
st.caption(f"{len(text):,} / {cfg['data']['max_chars_input']:,} characters")

if st.button("Analyse", type="primary"):
    try:
        validate_text(text, cfg["data"]["max_chars_input"])
        with st.spinner("Analysing..."):
            res = get_predictor(choice).predict(text)
    except InputError as e:
        st.error(str(e))
    except ModelNotAvailable as e:
        st.error(str(e))
    except Exception as e:                              # never crash the UI
        st.error(f"Something went wrong while predicting: {e}")
    else:
        word = "Real-like" if res.label == 1 else "Fake-like"
        conf = res.score_real if res.label == 1 else 1 - res.score_real
        st.subheader(f"Predicted dataset label: {word}")
        st.progress(float(conf), text=f"Model confidence: {conf:.1%}  (P(Real) = {res.score_real:.3f})")
        st.info(DISCLAIMER + " The score is model confidence, not a calibrated measure of truth.")
