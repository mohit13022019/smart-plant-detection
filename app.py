from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = r"C:\Users\mohit\OneDrive\Desktop\Python\SMART PLANT DETECTION\plant_pathology_dataset.csv"

df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()

REQUIRED_COLUMNS = [
    "Crop",
    "Disease",
    "Causal_organism",
    "Typical_symptoms",
    "Pathogen_group",
    "Spread_and_favourable_conditions",
    "Diagnostic_notes",
]

for column in REQUIRED_COLUMNS:
    if column not in df.columns:
        raise ValueError(f"Missing required dataset column: {column}")

df = df.dropna(subset=["Crop", "Disease", "Typical_symptoms"]).copy()

TEXT_COLUMNS = [
    "Crop",
    "Disease",
    "Causal_organism",
    "Typical_symptoms",
    "Pathogen_group",
    "Spread_and_favourable_conditions",
    "Diagnostic_notes",
]

for column in TEXT_COLUMNS:
    df[column] = df[column].fillna("").astype(str)

df["Disease"] = df["Disease"].str.strip()
df["Crop"] = df["Crop"].str.strip()

df["combined_text"] = (
    df["Crop"] + " " +
    df["Disease"] + " " +
    df["Causal_organism"] + " " +
    df["Typical_symptoms"] + " " +
    df["Pathogen_group"] + " " +
    df["Spread_and_favourable_conditions"] + " " +
    df["Diagnostic_notes"]
)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_tfidf = vectorizer.fit_transform(df["combined_text"])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Disease"])

MODELS = {
    "Naive Bayes": make_pipeline(
        TfidfVectorizer(stop_words="english", ngram_range=(1, 2)),
        MultinomialNB()
    ),
    "Logistic Regression": make_pipeline(
        TfidfVectorizer(stop_words="english", ngram_range=(1, 2)),
        LogisticRegression(max_iter=2000, class_weight="balanced")
    ),
    "SVM": make_pipeline(
        TfidfVectorizer(stop_words="english", ngram_range=(1, 2)),
        LinearSVC(class_weight="balanced", random_state=42)
    ),
    "Random Forest": make_pipeline(
        TfidfVectorizer(stop_words="english", ngram_range=(1, 2)),
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        )
    ),
}

for model in MODELS.values():
    model.fit(df["combined_text"], df["Disease"])


def get_management_suggestions(disease, pathogen_group):
    text = f"{disease} {pathogen_group}".lower()

    if any(word in text for word in ["fung", "mildew", "rust", "mould", "mold"]):
        return [
            "Remove and safely dispose of heavily infected plant material.",
            "Avoid overhead irrigation and prolonged leaf wetness.",
            "Improve plant spacing and airflow to reduce humidity.",
            "Consult local agricultural guidance before selecting a registered fungicide."
        ]

    if any(word in text for word in ["bacter"]):
        return [
            "Remove severely affected plant material using clean tools.",
            "Disinfect tools between plants and avoid handling wet foliage.",
            "Use clean planting material and avoid unnecessary plant injuries.",
            "Ask an agricultural specialist about locally approved control options."
        ]

    if any(word in text for word in ["virus", "viral"]):
        return [
            "Remove severely infected plants when recommended by local guidance.",
            "Inspect plants for insect vectors such as aphids or whiteflies.",
            "Use healthy planting material and control weeds that may host vectors.",
            "Choose resistant varieties when available."
        ]

    if any(word in text for word in ["nematode"]):
        return [
            "Inspect roots for galls, lesions, or unusual root damage.",
            "Rotate crops with suitable non-host crops where recommended.",
            "Clean soil and plant debris from tools and equipment.",
            "Consult an agricultural specialist before choosing a control treatment."
        ]

    return [
        "Remove severely damaged plant parts when appropriate.",
        "Avoid waterlogging and maintain suitable irrigation.",
        "Inspect nearby plants for similar symptoms.",
        "Confirm the diagnosis with an agricultural specialist before treatment."
    ]


def make_candidate(row, score):
    return {
        "crop": row["Crop"],
        "disease": row["Disease"],
        "score": round(float(score), 3),
        "causal_organism": row["Causal_organism"],
        "pathogen_group": row["Pathogen_group"],
        "symptoms": row["Typical_symptoms"],
        "spread": row["Spread_and_favourable_conditions"],
        "diagnostic_notes": row["Diagnostic_notes"],
        "suggestions": get_management_suggestions(
            row["Disease"], row["Pathogen_group"]
        ),
    }


def predict_diseases(crop, symptoms, method):
    query = f"{crop} {symptoms}".strip()

    if method == "TF-IDF + Cosine Similarity":
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, X_tfidf).flatten()

        candidates = df.copy()
        candidates["score"] = scores

        if crop and crop != "All crops":
            crop_mask = candidates["Crop"].str.casefold() == crop.casefold()
            if crop_mask.any():
                candidates = candidates[crop_mask]

        candidates = candidates.sort_values("score", ascending=False)
        candidates = candidates[candidates["score"] > 0]

        results = []
        seen = set()

        for _, row in candidates.iterrows():
            disease = row["Disease"]
            if disease in seen:
                continue
            seen.add(disease)
            results.append(make_candidate(row, row["score"]))
            if len(results) == 3:
                break

        return results

    model = MODELS[method]
    predictions = model.predict([query])

    if hasattr(model[-1], "predict_proba"):
        probabilities = model.predict_proba([query])[0]
        classes = model[-1].classes_
        ranked = sorted(
            zip(classes, probabilities),
            key=lambda item: item[1],
            reverse=True
        )[:3]
        ranked_diseases = [(name, score) for name, score in ranked]
    else:
        decision_scores = model.decision_function([query])[0]
        if np.isscalar(decision_scores):
            decision_scores = np.array([-decision_scores, decision_scores])

        classes = model[-1].classes_
        order = np.argsort(decision_scores)[::-1][:3]
        ranked_diseases = [
            (classes[index], float(decision_scores[index]))
            for index in order
        ]

    results = []

    for disease, score in ranked_diseases:
        matches = df[df["Disease"] == disease]

        if crop and crop != "All crops":
            crop_matches = matches[
                matches["Crop"].str.casefold() == crop.casefold()
            ]
            if not crop_matches.empty:
                matches = crop_matches

        if not matches.empty:
            results.append(make_candidate(matches.iloc[0], score))

    return results


@app.route("/", methods=["GET", "POST"])
def index():
    crops = sorted(df["Crop"].dropna().unique().tolist())
    methods = [
        "TF-IDF + Cosine Similarity",
        "Naive Bayes",
        "Logistic Regression",
        "SVM",
        "Random Forest",
    ]

    results = []
    error = None
    symptoms = ""
    selected_crop = "All crops"
    selected_method = "TF-IDF + Cosine Similarity"

    if request.method == "POST":
        symptoms = request.form.get("symptoms", "").strip()
        selected_crop = request.form.get("crop", "All crops")
        selected_method = request.form.get(
            "method", "TF-IDF + Cosine Similarity"
        )

        if not symptoms:
            error = "Please enter the plant symptoms."
        elif selected_method not in methods:
            error = "Please select a valid model."
        else:
            results = predict_diseases(
                selected_crop,
                symptoms,
                selected_method
            )

            if not results:
                error = (
                    "No suitable match was found. Try describing "
                    "the symptoms in more detail."
                )

    return render_template(
        "index.html",
        crops=crops,
        methods=methods,
        results=results,
        error=error,
        symptoms=symptoms,
        selected_crop=selected_crop,
        selected_method=selected_method,
    )


if __name__ == "__main__":
    app.run(debug=True)