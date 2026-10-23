"""Streamlit demo: paste an Urdu news article and get a real/fake prediction. Run with: streamlit run app.py"""
import json
import random
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from src.urdu_text import CORPUS_DIR, MODEL_PATH, download_corpus, normalize

ROOT = Path(__file__).parent
RESULTS = json.loads((ROOT / "results" / "summary.json").read_text())

st.set_page_config(page_title="Urdu Fake News Detector", page_icon="📰", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu&display=swap');
textarea, .urdu { direction: rtl; text-align: right; font-family: 'Noto Nastaliq Urdu', serif; font-size: 1.05rem; line-height: 2.1; }
.verdict { padding: 1rem 1.2rem; border-radius: 12px; font-size: 1.3rem; font-weight: 700; margin: .5rem 0 1rem; }
.fake { background: #fee4e2; color: #b42318; border: 1px solid #fda29b; }
.real { background: #e0eaff; color: #1d4ed8; border: 1px solid #a4bcfd; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Downloading sample articles...")
def sample_articles():
    """Test-set articles from the Bend The Truth corpus (downloaded from the authors' GitHub on first use)."""
    try:
        corpus = CORPUS_DIR if Path(CORPUS_DIR).exists() else download_corpus()
    except Exception:
        return []
    files = sorted((Path(corpus) / "Test").glob("*/*.txt"))
    return [(f.parent.name, f.name, f.read_text(encoding="utf-8", errors="ignore").strip()) for f in files]


def top_terms(model, text, n=8):
    """Words and phrases in this article that pushed the score towards fake or real."""
    est = model.calibrated_classifiers_[0].estimator
    vec, clf = est.named_steps["tfidf"], est.named_steps["clf"]
    if not hasattr(vec, "get_feature_names_out") or not hasattr(clf, "coef_"):
        return [], []
    x = vec.transform([normalize(text)])
    contrib = x.multiply(clf.coef_[0]).tocsr()
    idx, vals = contrib.indices, contrib.data
    names = vec.get_feature_names_out()
    order = np.argsort(vals)
    fake = [(names[idx[i]], vals[i]) for i in order[::-1][:n] if vals[i] > 0]
    real = [(names[idx[i]], vals[i]) for i in order[:n] if vals[i] < 0]
    return fake, real


with st.sidebar:
    st.markdown("## 📰 Urdu Fake News Detector")
    st.caption("TF-IDF + logistic regression trained on the *Bend The Truth* dataset")
    st.metric("Test accuracy", f"{RESULTS['test_accuracy']:.1%}")
    st.metric("ROC AUC", f"{RESULTS['test_roc_auc']:.3f}")
    st.markdown(
        "This model uses **normalised text**, so it can't rely on typing artifacts such as the Arabic yeh "
        f"(a one-character rule alone scores {RESULTS['one_character_rule_test_accuracy']:.0%} on this dataset). "
        "See the README for the full analysis."
    )

st.title("Urdu Fake News Detector")
st.write("Paste an Urdu news article, or load one from the test set, to see whether the model thinks it is real or fake.")
st.warning("Research demo trained on 638 articles. Treat predictions as a signal, not a verdict.", icon="⚠️")

samples = sample_articles()
st.session_state.setdefault("text", "")
st.session_state.setdefault("truth", None)
if samples:
    if st.button("🎲 Load a random test article"):
        label, name, text = random.choice(samples)
        st.session_state.text, st.session_state.truth = text, label
else:
    st.caption("Sample articles are unavailable right now. You can still paste your own text.")

text = st.text_area("Urdu news text", key="text", height=230, placeholder="یہاں خبر کا متن پیسٹ کریں...")

if text.strip():
    model = load_model()
    p_fake = float(model.predict_proba([normalize(text)])[0, 1])
    is_fake = p_fake >= 0.5
    css, label = ("fake", "Likely FAKE") if is_fake else ("real", "Likely REAL")
    st.markdown(f"<div class='verdict {css}'>{label} &nbsp;·&nbsp; {max(p_fake, 1 - p_fake):.0%} confidence</div>",
                unsafe_allow_html=True)
    st.progress(p_fake, text=f"Probability fake: {p_fake:.1%}")
    if st.session_state.truth:
        truth = st.session_state.truth.lower()
        st.caption(f"True label of this test article: **{truth.upper()}** "
                   f"{'✅' if (truth == 'fake') == is_fake else '❌'}")
    fake_terms, real_terms = top_terms(model, text)
    if fake_terms or real_terms:
        c1, c2 = st.columns(2)
        c1.markdown("**Pushed towards fake**")
        c1.markdown("<div class='urdu'>" + " · ".join(t for t, _ in fake_terms) + "</div>", unsafe_allow_html=True)
        c2.markdown("**Pushed towards real**")
        c2.markdown("<div class='urdu'>" + " · ".join(t for t, _ in real_terms) + "</div>", unsafe_allow_html=True)
