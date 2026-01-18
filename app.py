import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Interpretable Fake News Detection on Online Articles Using NLP",
    layout="centered"
)

st.title("📰 Interpretable Fake News Detection on Online Articles Using NLP")

st.write(
    """
    This application uses Natural Language Processing (NLP) and machine learning
    to classify online news articles as **Real** or **Fake**.

    The model is **interpretable**, meaning it not only provides predictions,
    but also explains **which words influenced the decision**.
    """
)

# ===============================
# Session State Initialization
# ===============================
if "title" not in st.session_state:
    st.session_state.title = ""

if "body" not in st.session_state:
    st.session_state.body = ""

if "show_result" not in st.session_state:
    st.session_state.show_result = False

# ===============================
# Load Models & Vectorizers
# ===============================
logreg_title = joblib.load("logreg_title.pkl")
tfidf_title = joblib.load("tfidf_title.pkl")

logreg_tb = joblib.load("logreg_title_body.pkl")
tfidf_tb = joblib.load("tfidf_title_body.pkl")

class_names = ["Real", "Fake"]

# ===============================
# Helper Functions
# ===============================
def predict(text, model, vectorizer):
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    return prediction, probability

def plot_lime_importance(exp):
    words, weights = zip(*exp.as_list())
    weights = np.array(weights)

    colors = ["red" if w > 0 else "green" for w in weights]

    fig, ax = plt.subplots()
    ax.barh(words, weights, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to Prediction")
    ax.set_title("Word Influence (Red → Fake | Green → Real)")
    plt.tight_layout()

    st.pyplot(fig)

def clear_all():
    st.session_state.title = ""
    st.session_state.body = ""
    st.session_state.show_result = False

# ===============================
# User Input
# ===============================
st.subheader("Enter News Article")

st.text_input(
    "News Title (required)",
    key="title"
)

st.text_area(
    "News Article Body (optional)",
    height=200,
    key="body"
)

# ===============================
# Buttons
# ===============================
col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("Predict")

with col2:
    st.button("Clear All", on_click=clear_all)

# ===============================
# Prediction Logic
# ===============================
if predict_btn:
    if st.session_state.title.strip() == "":
        st.warning("Please enter a news title.")
    else:
        st.session_state.show_result = True

        if st.session_state.body.strip() == "":
            combined_text = st.session_state.title
            model = logreg_title
            vectorizer = tfidf_title
            mode = "Title-only"
        else:
            combined_text = st.session_state.title + " " + st.session_state.body
            model = logreg_tb
            vectorizer = tfidf_tb
            mode = "Title + Body"

        pred, prob = predict(combined_text, model, vectorizer)

        st.markdown(f"### Prediction Mode: **{mode}**")
        st.markdown(
            f"### Prediction: **{'FAKE 🟥' if pred == 1 else 'REAL 🟩'}**"
        )

        st.write(
            f"Confidence — Real: {prob[0]*100:.2f}% | Fake: {prob[1]*100:.2f}%"
        )

        # ===============================
        # LIME Interpretability
        # ===============================
        explainer = LimeTextExplainer(class_names=class_names)

        exp = explainer.explain_instance(
            combined_text,
            lambda x: model.predict_proba(vectorizer.transform(x)),
            num_features=10
        )

        st.subheader("Most Influential Words")

        st.write(
            """
            **How to read this chart:**
            - **Green bars (left)** → words pushing the prediction towards **Real**
            - **Red bars (right)** → words pushing the prediction towards **Fake**
            - Longer bars indicate **stronger influence**
            """
        )

        plot_lime_importance(exp)