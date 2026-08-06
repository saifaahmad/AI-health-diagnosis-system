import os
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import spacy
from nltk.corpus import wordnet

nlp = spacy.load("en_core_web_sm")
# ---------- LOAD TRAINED MODELS FROM models/ FOLDER ----------

# Base directory: .../ai_health_diagnosis
# Paths

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DIABETES_MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_model.pkl")
HEART_MODEL_PATH    = os.path.join(BASE_DIR, "models", "heart_model.pkl")
SYMPTOM_MODEL_PATH = os.path.join(BASE_DIR, "models", "symptom_model.pkl")
SYMPTOM_ENCODER_PATH = os.path.join(BASE_DIR, "models", "symptom_label_encoder.pkl")

symptom_model = joblib.load(SYMPTOM_MODEL_PATH)
symptom_encoder = joblib.load(SYMPTOM_ENCODER_PATH)
diab_model = joblib.load(DIABETES_MODEL_PATH)
heart_model = joblib.load(HEART_MODEL_PATH)


# Simple keyword mapping for symptoms
symptom_keywords = {
    "fever": ["fever", "high temperature", "temperature", "feeling hot"],
    "cough": ["cough", "coughing"],
    "breathlessness": ["breathlessness", "shortness of breath", "difficulty breathing", "cant breathe", "unable to breathe properly"],
    "thirst": ["thirst", "excessive thirst"],
    "frequent urination": ["frequent urination", "pee a lot", "urinate a lot"],
    "blurred vision": ["blurred vision", "blurry vision"],
    "chest pain": ["chest pain", "pain in chest", "pressure in chest"],
    "shortness of breath": ["shortness of breath", "difficulty breathing", "breathlessness"],
    "fatigue": ["fatigue", "tired", "tiredness", "exhausted"],
    "headache": ["headache", "head pain"],
    "nausea": ["nausea", "feeling like vomiting", "sick to stomach"],
    "sensitivity to light": ["sensitivity to light", "light hurts eyes"],
    "joint pain": ["joint pain", "pain in joints"],
    "swelling": ["swelling", "swollen joints"],
    "morning stiffness": ["morning stiffness", "stiff in morning"]
}


# ---------- HEART DATA ENCODING MAPS (MUST MATCH TRAINING) ----------

cp_map = {
    "typical angina": 0,
    "atypical angina": 1,
    "non-anginal": 2,
    "asymptomatic": 3
}

restecg_map = {
    "normal": 0,
    "st-t abnormality": 1,
    "lv hypertrophy": 2
}

slope_map = {
    "upsloping": 0,
    "flat": 1,
    "downsloping": 2
}

thal_map = {
    "normal": 1,
    "fixed defect": 2,
    "reversable defect": 3
}

# ---------- STREAMLIT UI ----------

st.title("AI-Powered Health Diagnosis System")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Diabetes Prediction", "Heart Disease Prediction", "Symptom Chatbot", "About / Disclaimer"]
)


# ---------- DIABETES PAGE ----------
if page == "Diabetes Prediction":
    st.header("Diabetes Risk Prediction")

    preg = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose Level", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    insulin = st.number_input("Insulin Level", 0, 500, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.5)
    age = st.number_input("Age", 1, 120, 30)

    if st.button("Predict Diabetes"):
        input_data = [[preg, glucose, bp, skin, insulin, bmi, dpf, age]]
        result = diab_model.predict(input_data)

        if result[0] == 1:
            st.error("Model Prediction: Diabetes Detected (High Risk)")
        else:
            st.success("Model Prediction: No Diabetes Detected (Low Risk)")

        # SHAP Explainability
        st.subheader("Why this prediction?")
        feature_names = [
            "Pregnancies", "Glucose", "Blood Pressure",
            "Skin Thickness", "Insulin", "BMI",
            "Diabetes Pedigree", "Age"
        ]
        try:
            explainer = shap.TreeExplainer(diab_model)
            shap_input = np.array(input_data)
            shap_values = explainer.shap_values(shap_input)

            # Fix for SHAP 0.50.0
            sv = shap_values
            if hasattr(sv, 'values'):
                sv = sv.values
            sv = np.array(sv)
            if sv.ndim == 3:
                sv = sv[0, :, 1]
            elif sv.ndim == 2 and sv.shape[0] == 1:
                sv = sv[0]
            elif sv.ndim == 2:
                sv = sv[:, 1]
            else:
                sv = sv[0]
            sv = sv.flatten()

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#e74c3c' if v > 0 else '#3498db' for v in sv]
            bars = ax.barh(feature_names, sv, color=colors)
            ax.set_xlabel("SHAP Value (impact on prediction)")
            ax.set_title("Feature Contributions to Diabetes Prediction")
            ax.axvline(x=0, color='black', linewidth=0.8)
            for bar, val in zip(bars, sv):
                ax.text(
                    val + (0.001 if val >= 0 else -0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}',
                    va='center',
                    ha='left' if val >= 0 else 'right',
                    fontsize=9
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(
                "🔴 Red = increases diabetes risk  |  "
                "🔵 Blue = decreases diabetes risk"
            )
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")
    st.caption(
        "Disclaimer: This prediction is generated by a machine learning model "
        "trained on public datasets and is for academic demonstration only. "
        "It is NOT a medical diagnosis. Please consult a qualified doctor for "
        "any health-related decisions."
    )

# ---------- HEART PAGE ----------
elif page == "Heart Disease Prediction":
    st.header("Heart Disease Risk Prediction")

    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox(
        "Chest Pain Type",
        ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
    )
    trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["False", "True"])
    restecg = st.selectbox(
        "Resting ECG",
        ["normal", "st-t abnormality", "lv hypertrophy"]
    )
    thalch = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
    exang = st.selectbox("Exercise Induced Angina?", ["False", "True"])
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0)
    slope = st.selectbox(
        "Slope of ST Segment",
        ["upsloping", "flat", "downsloping"]
    )
    ca = st.number_input("Number of Major Vessels (0–3)", 0, 3, 0)
    thal = st.selectbox(
        "Thal",
        ["normal", "fixed defect", "reversable defect"]
    )

    if st.button("Predict Heart Disease"):
        sex_val = 1 if sex == "Male" else 0
        fbs_val = 1 if fbs == "True" else 0
        exang_val = 1 if exang == "True" else 0

        cp_val = cp_map[cp]
        restecg_val = restecg_map[restecg]
        slope_val = slope_map[slope]
        thal_val = thal_map[thal]

        input_data = [[
            age, sex_val, cp_val, trestbps, chol,
            fbs_val, restecg_val, thalch, exang_val,
            oldpeak, slope_val, ca, thal_val
        ]]

        result = heart_model.predict(input_data)

        if result[0] == 1:
            st.error("Model Prediction: Heart Disease Detected (High Risk)")
        else:
            st.success("Model Prediction: No Heart Disease Detected (Low Risk)")

        # SHAP Explainability
        st.subheader("Why this prediction?")
        feature_names = [
            "Age", "Sex", "Chest Pain Type",
            "Resting BP", "Cholesterol", "Fasting Blood Sugar",
            "Resting ECG", "Max Heart Rate", "Exercise Angina",
            "ST Depression", "ST Slope", "Major Vessels", "Thal"
        ]
        try:
            explainer = shap.TreeExplainer(heart_model)
            shap_input = np.array(input_data)
            shap_values = explainer.shap_values(shap_input)

            
           # Fix for SHAP 0.50.0
            sv = shap_values
            if hasattr(sv, 'values'):
                sv = sv.values
            sv = np.array(sv)
            if sv.ndim == 3:
                sv = sv[0, :, 1]
            elif sv.ndim == 2 and sv.shape[0] == 1:
                sv = sv[0]
            elif sv.ndim == 2:
                sv = sv[:, 1]
            else:
                sv = sv[0]
            sv = sv.flatten()

            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#e74c3c' if v > 0 else '#3498db' for v in sv]
            bars = ax.barh(feature_names, sv, color=colors)
            ax.set_xlabel("SHAP Value (impact on prediction)")
            ax.set_title("Feature Contributions to Heart Disease Prediction")
            ax.axvline(x=0, color='black', linewidth=0.8)
            for bar, val in zip(bars, sv):
                ax.text(
                    val + (0.001 if val >= 0 else -0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.3f}',
                    va='center',
                    ha='left' if val >= 0 else 'right',
                    fontsize=9
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.caption(
                "🔴 Red = increases heart disease risk  |  "
                "🔵 Blue = decreases heart disease risk"
            )
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")
    st.caption(
        "Disclaimer: This prediction is generated by a machine learning model "
        "trained on public datasets and is for academic demonstration only. "
        "It is NOT a medical diagnosis. Please consult a qualified doctor for "
        "any health-related decisions."
    )
elif page == "Symptom Chatbot":
    st.header("Symptom-Based Health Chatbot (NLP)")

    # Load and clean training data
    TRAINING_CSV_PATH = os.path.join(BASE_DIR, "data", "Training.csv")
    training_df = pd.read_csv(TRAINING_CSV_PATH)
    training_df = training_df.drop(columns=["Unnamed: 133"], errors="ignore")
    training_df["prognosis"] = training_df["prognosis"].str.strip()
    symptom_columns = [col for col in training_df.columns if col != "prognosis"]

    # Complete medical synonym map
    essential_synonyms = {
        # Fatigue
        "exhausted"             : "fatigue",
        "tired"                 : "fatigue",
        "tiredness"             : "fatigue",
        "lethargy"              : "fatigue",
        "feels tired"           : "fatigue",
        "feeling tired"         : "fatigue",
        # Fever
        "fever"                 : "high fever",
        "high temperature"      : "high fever",
        "temperature"           : "high fever",
        "feeling hot"           : "high fever",
        "low grade fever"       : "mild fever",
        "slight fever"          : "mild fever",
        # Breathing
        "breathless"            : "breathlessness",
        "cant breathe"          : "breathlessness",
        "cannot breathe"        : "breathlessness",
        "shortness of breath"   : "breathlessness",
        "difficulty breathing"  : "breathlessness",
        "short of breath"       : "breathlessness",
        # Cough
        "coughing"              : "cough",
        # Nausea/Vomiting
        "nauseous"              : "nausea",
        "feeling nauseous"      : "nausea",
        "throwing up"           : "vomiting",
        "puking"                : "vomiting",
        "threw up"              : "vomiting",
        # Pain
        "chest tightness"       : "chest pain",
        "chest hurts"           : "chest pain",
        "heart pain"            : "chest pain",
        "stomach ache"          : "stomach pain",
        "tummy ache"            : "stomach pain",
        "belly pain"            : "abdominal pain",
        "belly ache"            : "abdominal pain",
        "back ache"             : "back pain",
        "backache"              : "back pain",
        "neck ache"             : "neck pain",
        "knee ache"             : "knee pain",
        "joint ache"            : "joint pain",
        "muscle ache"           : "muscle pain",
        "head ache"             : "headache",
        # Skin
        "itchy"                 : "itching",
        "itchy skin"            : "itching",
        "rash"                  : "skin rash",
        "skin peeling"          : "skin peeling",
        # Eyes/Vision
        "blurry vision"         : "blurred and distorted vision",
        "blurred vision"        : "blurred and distorted vision",
        "yellow eyes"           : "yellowing of eyes",
        # Other
        "dizzy"                 : "dizziness",
        "feeling dizzy"         : "dizziness",
        "thirsty"               : "thirst",
        "feeling thirsty"       : "thirst",
        "very thirsty"          : "thirst",
        "no appetite"           : "loss of appetite",
        "not hungry"            : "loss of appetite",
        "yellow skin"           : "yellowish skin",
        "dark pee"              : "dark urine",
        "peeing a lot"          : "frequent urination",
        "urinating a lot"       : "frequent urination",
        "heart racing"          : "fast heart rate",
        "palpitations"          : "fast heart rate",
        "sweaty"                : "sweating",
        "chilly"                : "chills",
        "shaky"                 : "shivering",
        "weight gain"           : "weight gain",
        "gaining weight"        : "weight gain",
        "losing weight"         : "weight loss",
        "anxious"               : "anxiety",
        "depressed"             : "depression",
        "stiff neck"            : "stiff neck",
        "swollen"               : "swelling",
        "bloated"               : "swelling of stomach",
    }

    # Disease words to filter out from input
    disease_words = [
        "diabetes", "cancer", "asthma", "arthritis",
        "hypertension", "migraine", "tuberculosis",
        "dengue", "malaria", "typhoid", "pneumonia"
    ]

    user_input = st.text_input(
        "Describe your symptoms in plain English",
        placeholder="e.g. I have chest pain and feel very exhausted"
    )

    if st.button("Check Possible Condition"):
        if not user_input.strip():
            st.info("Please enter your symptoms first.")
        else:
            # Step 1: lowercase
            modified_input = user_input.lower()

            # Step 2: remove disease names
            for word in disease_words:
                modified_input = modified_input.replace(word, "")

            # Step 3: apply synonyms (longest first to avoid partial replacements)
            for phrase in sorted(essential_synonyms.keys(), key=len, reverse=True):
                modified_input = modified_input.replace(
                    phrase, essential_synonyms[phrase]
                )

            # Step 4: NLP tokenization
            doc = nlp(modified_input)
            user_tokens = set()
            for token in doc:
                if not token.is_stop and not token.is_punct:
                    user_tokens.add(token.lemma_)

            # Step 5: Match against symptom columns
            matched_symptoms = []
            for symptom in symptom_columns:
                symptom_words = symptom.strip().lower().replace("_", " ")
                symptom_doc = nlp(symptom_words)
                symptom_tokens = set(
                    token.lemma_ for token in symptom_doc
                    if not token.is_stop and not token.is_punct
                )
                if symptom_tokens and symptom_tokens.issubset(user_tokens):
                    matched_symptoms.append(symptom)

            matched_symptoms = list(set(matched_symptoms))

            # Step 6: Predict
            if len(matched_symptoms) >= 2:
                input_vector = [
                    1 if s in matched_symptoms else 0
                    for s in symptom_columns
                ]

                proba = symptom_model.predict_proba([input_vector])[0]
                top3_idx = proba.argsort()[-3:][::-1]
                top3_diseases = symptom_encoder.inverse_transform(top3_idx)
                top3_conf = [round(proba[i] * 100, 1) for i in top3_idx]

                st.warning("**Top Possible Conditions:**")
                icons = ["1️⃣", "2️⃣", "3️⃣"]
                funcs = [st.error, st.warning, st.info]
                for i in range(3):
                    funcs[i](
                        f"{icons[i]} **{top3_diseases[i]}** "
                        f"— {top3_conf[i]}% confidence"
                    )

                st.info(
                    f"**Symptoms detected:** "
                    f"{', '.join([s.replace('_', ' ') for s in matched_symptoms])}"
                )

            elif len(matched_symptoms) == 1:
                st.info(
                    f"Only **{matched_symptoms[0].replace('_',' ')}** "
                    f"matched — not enough to suggest a condition. "
                    f"Please describe more symptoms or consult a doctor."
                )
            else:
                st.info(
                    "No matching symptoms found. Try describing "
                    "differently or consult a doctor."
                )

    st.caption(
        "Disclaimer: This chatbot uses NLP + ML on a medical symptom "
        "dataset for academic demonstration only. Not a medical diagnosis."
    )
# ---------- ABOUT / DISCLAIMER PAGE ----------
elif page == "About / Disclaimer":
    st.header("About the Project")

    st.write(
        """
        This AI-Powered Health Diagnosis System is a final-year B.Tech project
        developed to demonstrate how machine learning can be applied to
        healthcare risk prediction.

        **Modules implemented:**
        - Diabetes risk prediction using the Pima Indians Diabetes dataset.
        - Heart disease risk prediction using the UCI Heart Disease dataset.
        - Web-based user interface using Streamlit for easy interaction.

        **Machine Learning Techniques:**
        - Random Forest classifiers for tabular clinical data.
        - Basic feature encoding and preprocessing for categorical heart disease features.
        """
    )

    st.subheader("Important Disclaimer")
    st.write(
        """
        This system is an academic prototype and is **not** a certified medical
        device. The predictions are based on public datasets and simplified
        ML models. The results **must not** be used for real medical decisions.

        For any health concerns, symptoms, or diagnosis, always consult a
        qualified healthcare professional.
        """
    )

    st.info(
        "This tool is intended only for learning, research demonstration, and "
        "academic evaluation."
    )
