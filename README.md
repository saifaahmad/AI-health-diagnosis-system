\# AI-Powered Health Diagnosis System



A machine learning-based health diagnosis system that predicts possible diseases from user-described symptoms, with explainable AI to show \*why\* each prediction was made.



\## Features

\- \*\*Symptom-based disease prediction\*\* using a Random Forest classifier trained on a symptom-disease dataset

\- \*\*NLP-based symptom input processing\*\* using spaCy, with a curated synonym mapping to normalize free-text user input into recognizable symptoms

\- \*\*Explainable AI (XAI)\*\* via SHAP (Shapley Additive Explanations) to surface feature-level contributions behind each prediction, improving transparency and trust

\- \*\*Interactive web interface\*\* built with Streamlit for real-time symptom entry and diagnosis



\## Tech Stack

\- Python

\- Scikit-learn (Random Forest)

\- spaCy (NLP preprocessing)

\- SHAP (model explainability)

\- Streamlit (deployment/UI)



\## Project Structure

├── app/ # Streamlit application files

├── data/ # Training and testing datasets

├── models/ # Trained model files (.pkl)

├── notebooks/ # Development/experimentation notebooks



\## How to Run

```bash

pip install -r requirements.txt

streamlit run app/chatbot\_app.py

```



\## How It Works

1\. User describes symptoms in natural language

2\. spaCy-based NLP pipeline normalizes the input against a curated symptom vocabulary

3\. Random Forest model predicts the most likely disease

4\. SHAP explains which symptoms most influenced the prediction

5\. Results are displayed through an interactive Streamlit interface



\## Limitations

This is an academic/portfolio project and is \*\*not intended for real medical diagnosis\*\*. Predictions are based on a limited training dataset and should not replace professional medical advice.

