
# ==========================================================
# SMS SPAM DETECTION USING MACHINE LEARNING AND ANN
# Academic Project
# ==========================================================

import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
h1 {
    color: #17365d;
}
div.stButton > button {
    background-color: #1769aa;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PROJECT TITLE ----------------

st.title("📩 SMS Spam Detection System")

st.subheader(
    "Using Machine Learning and Artificial Neural Network"
)

st.write(
    "This application detects whether an SMS message "
    "is genuine or spam using trained classification models."
)

st.divider()

# ---------------- DATASET PATH ----------------

DATA_FILE = "SMSSpamCollection.txt"


# ---------------- LOAD AND TRAIN MODELS ----------------

@st.cache_resource
def train_models():

    # Check dataset
    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError(
            "SMSSpamCollection.txt not found. "
            "Put it in the same folder as app.py."
        )

    # Load tab-separated dataset
    data = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )

    # Clean dataset
    data = data.dropna(subset=["label", "message"])

    data["label"] = (
        data["label"].astype(str).str.strip().str.lower()
    )

    data["message"] = data["message"].astype(str)

    data = data[
        data["label"].isin(["ham", "spam"])
    ]

    data = data.drop_duplicates(
        subset=["label", "message"]
    )

    if data.empty or data["label"].nunique() != 2:
        raise ValueError(
            "The dataset must contain ham and spam labels."
        )

    # Encode labels: ham = 0, spam = 1
    data["target"] = data["label"].map({
        "ham": 0,
        "spam": 1
    })

    # Split training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        data["message"],
        data["target"],
        test_size=0.20,
        random_state=42,
        stratify=data["target"]
    )

    # Convert text into numerical features
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_test_tfidf = vectorizer.transform(X_test)

    # ---------------- MODEL 1: MACHINE LEARNING ----------------

    nb_model = MultinomialNB(alpha=0.5)

    nb_model.fit(
        X_train_tfidf,
        y_train
    )

    # ---------------- MODEL 2: ANN ----------------
    # MLPClassifier is a feed-forward neural network.

    ann_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=30,
        early_stopping=True,
        random_state=42
    )

    ann_model.fit(
        X_train_tfidf,
        y_train
    )

    # Store both models
    models = {
        "Naive Bayes (ML)": nb_model,
        "Artificial Neural Network (ANN)": ann_model
    }

    # ---------------- MODEL EVALUATION ----------------

    results = []
    predictions = {}

    for name, model in models.items():

        y_pred = model.predict(X_test_tfidf)

        predictions[name] = y_pred

        results.append({
            "Model": name,

            "Accuracy (%)": round(
                accuracy_score(y_test, y_pred) * 100, 2
            ),

            "Precision (%)": round(
                precision_score(
                    y_test, y_pred, zero_division=0
                ) * 100, 2
            ),

            "Recall (%)": round(
                recall_score(
                    y_test, y_pred, zero_division=0
                ) * 100, 2
            ),

            "F1 Score (%)": round(
                f1_score(
                    y_test, y_pred, zero_division=0
                ) * 100, 2
            )
        })

    return (
        data,
        vectorizer,
        models,
        X_test_tfidf,
        y_test,
        results,
        predictions
    )


# ---------------- START APPLICATION ----------------

try:
    (
        data,
        vectorizer,
        models,
        X_test_tfidf,
        y_test,
        results,
        predictions
    ) = train_models()

except Exception as error:
    st.error(f"Project error: {error}")
    st.info(
        "Check the dataset filename, dataset format, "
        "and installed Python libraries."
    )
    st.stop()


# ---------------- DASHBOARD ----------------

st.success("Dataset loaded and models trained successfully.")

total_messages = len(data)

spam_count = int(
    (data["label"] == "spam").sum()
)

ham_count = int(
    (data["label"] == "ham").sum()
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Messages",
    f"{total_messages:,}"
)

col2.metric(
    "Spam Messages",
    f"{spam_count:,}"
)

col3.metric(
    "Genuine Messages",
    f"{ham_count:,}"
)

st.divider()


# ---------------- SMS PREDICTION ----------------

st.header("🔍 Test Your SMS Message")

message = st.text_area(
    "Enter your SMS message:",
    placeholder=(
        "Example: Congratulations! You won a free prize. "
        "Call now to claim your reward."
    ),
    height=120
)

selected_model = st.selectbox(
    "Choose a model:",
    list(models.keys())
)

if st.button("Analyze SMS Message"):

    if not message.strip():

        st.warning(
            "Please enter a message before analyzing."
        )

    else:

        message_tfidf = vectorizer.transform([message])

        model = models[selected_model]

        prediction = int(
            model.predict(message_tfidf)[0]
        )

        probabilities = model.predict_proba(
            message_tfidf
        )[0]

        spam_probability = float(
            probabilities[1] * 100
        )

        if prediction == 1:

            st.error("🚨 SPAM MESSAGE DETECTED")

            st.write(
                "The selected model classified this SMS as spam."
            )

        else:

            st.success("✅ GENUINE MESSAGE (HAM)")

            st.write(
                "The selected model classified this SMS as genuine."
            )

        st.progress(
            int(round(spam_probability)),
            text=(
                f"Estimated spam probability: "
                f"{spam_probability:.2f}%"
            )
        )

        st.caption(
            "Predictions are estimates and may not always be correct."
        )


# ---------------- PERFORMANCE TABLE ----------------

st.divider()

st.header("📊 Model Performance Comparison")

results_df = pd.DataFrame(results)

st.dataframe(
    results_df,
    use_container_width=True,
    hide_index=True
)

best_model = max(
    results,
    key=lambda item: item["F1 Score (%)"]
)

st.info(
    f"Best model by F1 score on the test set: "
    f"{best_model['Model']} "
    f"({best_model['F1 Score (%)']}%)"
)


# ---------------- ACCURACY CHART ----------------

st.subheader("📈 Accuracy Comparison")

fig, ax = plt.subplots(figsize=(8, 4))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy (%)",
    hue="Model",
    legend=False,
    ax=ax
)

ax.set_ylim(0, 100)

ax.set_xlabel("Model")

ax.set_ylabel("Accuracy (%)")

ax.set_title("Machine Learning vs ANN")

ax.tick_params(
    axis="x",
    rotation=10
)

plt.tight_layout()

st.pyplot(fig)

plt.close(fig)


# ---------------- CONFUSION MATRIX ----------------

st.divider()

st.subheader("🧮 Confusion Matrix")

evaluation_model = st.selectbox(
    "Select model for confusion matrix:",
    list(models.keys()),
    key="evaluation_model"
)

y_pred = predictions[evaluation_model]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

fig2, ax2 = plt.subplots(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Genuine", "Spam"],
    yticklabels=["Genuine", "Spam"],
    ax=ax2
)

ax2.set_xlabel("Predicted Label")

ax2.set_ylabel("Actual Label")

ax2.set_title("Confusion Matrix")

plt.tight_layout()

st.pyplot(fig2)

plt.close(fig2)


# ---------------- CLASSIFICATION REPORT ----------------

with st.expander("View Detailed Classification Report"):

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Genuine (ham)", "Spam"],
        zero_division=0
    )

    st.text(report)


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Academic Project | Python | TF-IDF | "
    "Naive Bayes | ANN | Scikit-learn | Streamlit"
)
