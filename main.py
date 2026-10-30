# ============================================================
# SMART PLANT DISEASE DETECTION
# Symptom-Based Analysis Using 5 Models / Methods
# ============================================================

# STEP 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


# ============================================================
# STEP 2: LOAD DATASET
# ============================================================

file_path = "plant_pathology_dataset.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET INFORMATION ==========")

print("Dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types and non-null counts:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nNumber of unique crops:")
print(df["Crop"].nunique())

print("\nNumber of unique diseases:")
print(df["Disease"].nunique())

print("\nNumber of records per crop:")
print(df["Crop"].value_counts())

print("\nNumber of records per pathogen group:")
print(df["Pathogen_group"].value_counts())


# ============================================================
# STEP 3: DATA CLEANING
# ============================================================

# Remove duplicate records, if any.
df = df.drop_duplicates().reset_index(drop=True)

# Remove rows missing essential fields.
df = df.dropna(
    subset=["Crop", "Disease", "Typical_symptoms"]
).reset_index(drop=True)

# Standardise crop and disease names.
df["Crop"] = df["Crop"].astype(str).str.strip()

df["Disease"] = df["Disease"].astype(str).str.strip()

# Fill missing descriptive information.
text_columns = [
    "Causal_organism",
    "Typical_symptoms",
    "Pathogen_group",
    "Spread_and_favourable_conditions",
    "Diagnostic_notes"
]

for col in text_columns:
    df[col] = df[col].fillna("").astype(str).str.strip()

# Create a normalised crop field for filtering.
df["Crop_normalized"] = df["Crop"].str.casefold()

# Build the text used for disease matching.
# Crop information is included to provide context.
df["Model_text"] = (
    "Crop: " + df["Crop"] + ". " +
    "Symptoms: " + df["Typical_symptoms"]
)

print("\n========== CLEANED DATA ==========")

print("Rows after cleaning:", len(df))
print("Unique crops:", df["Crop"].nunique())
print("Unique diseases:", df["Disease"].nunique())


# ============================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# 4A. Distribution of crops

plt.figure(figsize=(12, 6))

df["Crop"].value_counts().plot(kind="bar")

plt.title("Number of Disease Records per Crop")
plt.xlabel("Crop")
plt.ylabel("Number of Records")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# 4B. Distribution of pathogen groups

plt.figure(figsize=(8, 5))

df["Pathogen_group"].value_counts().plot(kind="bar")

plt.title("Distribution of Pathogen Groups")
plt.xlabel("Pathogen Group")
plt.ylabel("Number of Records")

plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()


# 4C. Number of records per disease

disease_counts = df["Disease"].value_counts()

print("\nDisease frequency distribution:")
print(disease_counts.value_counts().sort_index())

plt.figure(figsize=(10, 5))

disease_counts.value_counts().sort_index().plot(kind="bar")

plt.title("Frequency of Disease Labels")
plt.xlabel("Number of Records per Disease")
plt.ylabel("Number of Disease Labels")

plt.tight_layout()
plt.show()


# ============================================================
# STEP 5: TEXT PREPROCESSING AND TF-IDF
# ============================================================

# TF-IDF converts text descriptions into numerical features.
#
# Unigrams: individual words
# Bigrams: pairs of consecutive words

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

# Learn vocabulary and transform all disease descriptions.
X = vectorizer.fit_transform(df["Model_text"])

# Target labels for supervised classification.
y = df["Disease"].to_numpy()

print("\n========== TF-IDF INFORMATION ==========")

print("TF-IDF matrix shape:", X.shape)

print("Number of text features:", len(vectorizer.vocabulary_))


# ============================================================
# STEP 6: MODEL 1 - TF-IDF + COSINE SIMILARITY
# ============================================================

# This is an information-retrieval method, not a supervised
# disease classifier.
#
# It compares a user's symptoms with the reference descriptions.

def cosine_disease_search(crop, symptoms, top_n=3):

    crop_key = crop.strip().casefold()

    # Filter reference records by crop.
    candidate_indices = np.flatnonzero(
        df["Crop_normalized"].to_numpy() == crop_key
    )

    if len(candidate_indices) == 0:
        print("Crop not found in the dataset.")
        return pd.DataFrame()

    # Convert the user's crop and symptoms into TF-IDF features.
    query_text = "Crop: " + crop.strip() + ". Symptoms: " + symptoms

    query_vector = vectorizer.transform([query_text])

    # Compare the query against descriptions of the selected crop.
    candidate_vectors = X[candidate_indices]

    similarities = cosine_similarity(
        query_vector,
        candidate_vectors
    ).ravel()

    # Rank descriptions by similarity.
    ranked_positions = np.argsort(similarities)[::-1][:top_n]

    selected_indices = candidate_indices[ranked_positions]

    results = df.iloc[selected_indices].copy()

    results["Score"] = similarities[ranked_positions]

    return results


# ============================================================
# STEP 7: TRAIN THE FOUR SUPERVISED MODELS
# ============================================================

# IMPORTANT:
# The dataset contains 41 records and approximately 40 disease
# labels. Most diseases have only one record.
#
# The models below can be fitted, but reliable generalisation
# cannot be established from this dataset.
#
# These are demonstration fits, not validated diagnostic models.

models = {

    # MODEL 2: NAIVE BAYES
    "Naive Bayes": MultinomialNB(),

    # MODEL 3: LOGISTIC REGRESSION
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ),

    # MODEL 4: LINEAR SUPPORT VECTOR MACHINE
    "SVM": LinearSVC(
        class_weight="balanced",
        random_state=42
    ),

    # MODEL 5: RANDOM FOREST
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )
}


# Fit each model to the complete dataset.
#
# We deliberately do not report a train/test accuracy here:
# a meaningful independent test would require more examples
# for the disease classes.

print("\n========== FITTING SUPERVISED MODELS ==========")

for model_name, model in models.items():

    model.fit(X, y)

    print(f"{model_name}: fitted successfully")

print(
    "\nWARNING: These fits demonstrate model implementation. "
    "They do not establish diagnostic accuracy."
)


# ============================================================
# STEP 8: RANK DISEASES USING A SUPERVISED MODEL
# ============================================================

def supervised_disease_search(
    model_name,
    crop,
    symptoms,
    top_n=3
):

    crop_key = crop.strip().casefold()

    # Filter the reference dataset by crop.
    candidate_indices = np.flatnonzero(
        df["Crop_normalized"].to_numpy() == crop_key
    )

    if len(candidate_indices) == 0:
        print("Crop not found in the dataset.")
        return pd.DataFrame()

    model = models[model_name]

    # Construct the query using the same format as the training text.
    query_text = "Crop: " + crop.strip() + ". Symptoms: " + symptoms

    query_vector = vectorizer.transform([query_text])

    # Obtain scores for the classes learned by the model.
    if hasattr(model, "predict_proba"):

        all_scores = model.predict_proba(query_vector)[0]

    elif hasattr(model, "decision_function"):

        all_scores = model.decision_function(query_vector)

        # LinearSVC returns a one-dimensional score when only
        # two classes exist. Handle that case defensively.
        if all_scores.ndim == 0:
            all_scores = np.array([all_scores])

    else:
        raise ValueError("The selected model does not provide scores.")

    # Map disease names to model output scores.
    model_classes = model.classes_

    class_score_map = {
        label: float(score)
        for label, score in zip(model_classes, all_scores)
    }

    # Keep only diseases belonging to the selected crop.
    candidates = df.iloc[candidate_indices].copy()

    candidates["Score"] = candidates["Disease"].map(
        class_score_map
    )

    candidates = candidates.dropna(subset=["Score"])

    # Rank candidate diseases by their model scores.
    results = candidates.sort_values(
        "Score",
        ascending=False
    ).head(top_n)

    return results


# ============================================================
# STEP 9: DISPLAY DISEASE INFORMATION
# ============================================================

def display_results(results, model_name):

    if results.empty:
        print("No matching crop or disease records were found.")
        return

    print("\n========== POSSIBLE DISEASE MATCHES ==========")

    print("Method:", model_name)

    for rank, (_, row) in enumerate(results.iterrows(), start=1):

        print(f"\n--- Candidate {rank}: {row['Disease']} ---")

        print("Crop:", row["Crop"])

        print("Causal organism:", row["Causal_organism"])

        print("Pathogen group:", row["Pathogen_group"])

        print("Typical symptoms:", row["Typical_symptoms"])

        print(
            "Favourable conditions:",
            row["Spread_and_favourable_conditions"]
        )

        print("Diagnostic notes:", row["Diagnostic_notes"])

        print("Model score:", round(float(row["Score"]), 4))

    print(
        "\nIMPORTANT: These are candidate matches, not confirmed "
        "diagnoses. Similarity and classifier scores are not "
        "calibrated disease probabilities."
    )


# ============================================================
# STEP 10: INTERACTIVE DISEASE SEARCH
# ============================================================

def run_prediction():

    print("\n==========================================")
    print(" SMART PLANT DISEASE SCREENING")
    print("==========================================")

    print("\nAvailable crops:")

    available_crops = sorted(df["Crop"].unique())

    for crop_name in available_crops:
        print("-", crop_name)

    crop = input("\nEnter the affected crop: ").strip()

    # Validate crop input.
    matching_crops = {
        crop_name.casefold(): crop_name
        for crop_name in available_crops
    }

    if crop.casefold() not in matching_crops:
        print("Crop not found. Please check the spelling.")
        return

    crop = matching_crops[crop.casefold()]

    symptoms = input(
        "\nDescribe the observed plant symptoms: "
    ).strip()

    if not symptoms:
        print("Please enter at least one symptom.")
        return

    print("\nChoose a method:")

    print("1. TF-IDF + Cosine Similarity")
    print("2. Naive Bayes")
    print("3. Logistic Regression")
    print("4. SVM")
    print("5. Random Forest")

    choice = input("\nEnter choice (1-5): ").strip()

    method_map = {
        "1": "TF-IDF + Cosine Similarity",
        "2": "Naive Bayes",
        "3": "Logistic Regression",
        "4": "SVM",
        "5": "Random Forest"
    }

    if choice not in method_map:
        print("Invalid choice.")
        return

    selected_method = method_map[choice]

    if choice == "1":

        results = cosine_disease_search(
            crop=crop,
            symptoms=symptoms,
            top_n=3
        )

    else:

        results = supervised_disease_search(
            model_name=selected_method,
            crop=crop,
            symptoms=symptoms,
            top_n=3
        )

    display_results(results, selected_method)


# ============================================================
# STEP 11: RUN THE INTERACTIVE PROGRAM
# ============================================================

if __name__ == "__main__":

    while True:

        run_prediction()

        again = input(
            "\nDo you want to test another case? (yes/no): "
        ).strip().lower()

        if again not in ["yes", "y"]:
            print("\nProgram finished.")
            break