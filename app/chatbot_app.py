# # NOTE: This is the initial prototype version of the symptom chatbot.
# # The fully upgraded NLP + ML version is integrated in diabetes_app.py
# # This file is kept for project documentation and version history purposes.
# import streamlit as st
# import pandas as pd
# import os
# st.title("Symptom-Based Health Chatbot")

# # Load symptom dataset

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# symptoms_df = pd.read_csv(os.path.join(BASE_DIR, "data", "symptoms.csv"))
# # User input
# user_input = st.text_input("Describe your symptoms:")

# doc = nlp(user_input.lower())

# # Extract meaningful tokens - nouns, adjectives, verbs
# user_tokens = set()
# for token in doc:
#     if not token.is_stop and not token.is_punct:
#         user_tokens.add(token.lemma_)

# best_match = None
# best_score = 0

# for _, row in symptoms_df.iterrows():
#     disease = row["disease"]
#     score = 0

#     for symptom_col in ["symptom1", "symptom2", "symptom3"]:
#         symptom = row[symptom_col]
#         if pd.isna(symptom):
#             continue

#         # NLP process the symptom
#         symptom_doc = nlp(symptom.strip().lower())
#         symptom_tokens = set(
#             token.lemma_ for token in symptom_doc
#             if not token.is_stop and not token.is_punct
#         )

#         # Check overlap between user tokens and symptom tokens
#         overlap = user_tokens & symptom_tokens
#         if overlap:
#             score += 1

#     if score > best_score:
#         best_score = score
#         best_match = disease