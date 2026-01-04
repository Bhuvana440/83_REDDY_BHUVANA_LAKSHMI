import pickle
import pandas as pd
import numpy as np
import streamlit as st

# -------------------------------
# Load trained model and features
# -------------------------------
model = pickle.load(open("model.pkl", "rb"))
trained_features = pickle.load(open("features.pkl", "rb"))

# -------------------------------
# Load precautions dataset
# -------------------------------
precaution_df = pd.read_csv("data/Disease precaution.csv")

def get_precautions(disease_name):
    row = precaution_df[precaution_df["Disease"] == disease_name]
    if row.empty:
        return []
    return row.iloc[0, 1:].dropna().tolist()

# -------------------------------
# Prediction function (with guardrails)
# -------------------------------
def predict_from_symptoms(input_symptoms, top_k=3):
    input_vector = pd.Series(0, index=trained_features)
    unknown_symptoms = []

    for symptom in input_symptoms:
        if symptom in input_vector.index:
            input_vector[symptom] = 1
        else:
            unknown_symptoms.append(symptom)

    input_df = pd.DataFrame([input_vector])
    proba = model.predict_proba(input_df)[0]
    classes = model.classes_

    top_idx = np.argsort(proba)[-top_k:][::-1]
    results = [(classes[i], proba[i]) for i in top_idx]

    return results, unknown_symptoms

# -------------------------------
# Explanation generator
# -------------------------------
def generate_explanation(symptoms, disease, confidence):
    return (
        f"The condition **{disease}** is suggested because it is commonly "
        f"associated with symptoms such as {', '.join(symptoms[:2])}. "
        f"The confidence score indicates the model’s likelihood based on "
        f"learned patterns from the dataset."
    )

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="AI Symptom Checker", layout="centered")

st.title("🩺 AI-Based Symptom Checker")
st.write("Select symptoms or enter custom symptoms to get top-3 possible conditions.")

st.divider()

# 🔹 MULTI-SELECT INPUT (KNOWN SYMPTOMS)
selected_symptoms = st.multiselect(
    "Select symptoms from the list:",
    options=sorted(trained_features)
)

# 🔹 FREE-TEXT INPUT (OPTIONAL)
user_input = st.text_input(
    "Add any other symptoms (comma-separated, optional):",
    ""
)

# Combine both inputs
custom_symptoms = []
if user_input.strip():
    custom_symptoms = [s.strip() for s in user_input.split(",")]

all_symptoms = list(set(selected_symptoms + custom_symptoms))

if st.button("Predict Disease"):
    if not all_symptoms:
        st.error("Please select or enter at least one symptom.")
    else:
        results, unknown_symptoms = predict_from_symptoms(all_symptoms)

        # Guardrail warning
        if unknown_symptoms:
            st.warning(
                f"⚠️ These symptoms are not recognized by the model and were ignored: "
                f"{', '.join(unknown_symptoms)}"
            )

        st.subheader("🔍 Top-3 Predicted Diseases")

        for disease, confidence in results:
            st.markdown(f"### {disease}")
            st.write(f"**Confidence:** {confidence:.2f}")

            explanation = generate_explanation(all_symptoms, disease, confidence)
            st.info(explanation)

            precautions = get_precautions(disease)
            if precautions:
                st.write("**Recommended Precautions:**")
                for p in precautions:
                    st.write(f"- {p}")

            st.divider()

st.warning(
    "⚠️ This application is for educational purposes only and does NOT "
    "provide medical diagnosis. Always consult a qualified healthcare professional."
)
