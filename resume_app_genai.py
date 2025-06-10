import numpy as np
import pickle
import re
import nltk
import fitz  # PyMuPDF
import openai
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import streamlit as st
# 📦 Setup
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 🧹 Clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    return ' '.join([lemmatizer.lemmatize(word) for word in text if word not in stop_words])

# 📄 Extract PDF text
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return " ".join([page.get_text() for page in doc])
    
import google.generativeai as genai
import os
# 🔑 Load Gemini API key
api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

# ✨ Gemini Functions
def validate_role_with_gemini(resume_text, role_options):
    prompt = f"""
You are an AI career assistant. Based on the resume text below, which of the following roles best matches the profile?
Options: {', '.join(role_options)}

Resume:
{resume_text}
"""
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text.strip()

def get_resume_feedback_gemini(resume_text, target_role):
    prompt = f"""
You are a resume coach. Analyze the following resume for a '{target_role}' role.
Suggest improvements in bullet points, including structure, tone, and missing skills.

Resume:
{resume_text}
"""
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text

# Load model and label encoder
MODEL_PATH = "resume_classifier_ml.pkl"
ENCODER_PATH = "label_encoder.pickle"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# 🚀 Streamlit Config
st.set_page_config(page_title="Resume Optimizer", page_icon="🧠", layout="wide")
st.sidebar.title("📄 Resume Optimizer")
st.sidebar.markdown("""
Upload your resume and receive:
- 🔍 ML Role Predictions
- 💡 Gemini AI Suggestions
---
Built by **Shaurya Chauhan**
[GitHub](https://github.com/Shaurya2127) | [LinkedIn](https://www.linkedin.com/in/shaurya-chauhan-0089911bb/)
""")

# Header
st.title("🧠 Resume Analyzer & Optimizer")
st.markdown("<h5 style='color: gray;'>Get job-fit predictions and GPT-based resume suggestions</h5>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📊 ML Role Prediction", "💡 Gemini Resume Feedback"])

# --------------------------- TAB 1 --------------------------------
with tab1:
    st.header("📊 Predict Your Resume's Role using ML")
    uploaded_file = st.file_uploader("📤 Upload your resume (PDF)", type="pdf", key="ml_tab")

    if uploaded_file:
        resume_text_ml = extract_text_from_pdf(uploaded_file)
        st.text_area("📄 Extracted Resume Text", resume_text_ml, height=250, key="resume_text_ml_box")

        def predict_resume(text, confidence_threshold=0.15):
            cleaned = clean_text(text)
            proba = model.predict_proba([cleaned])[0]
            top_idx = np.argsort(proba)[::-1][:3]
            top_labels = label_encoder.inverse_transform(top_idx)
            top_scores = proba[top_idx]
            return [(label, score) for label, score in zip(top_labels, top_scores) if score >= confidence_threshold]

        if st.button("🔍 Predict Job Role"):
            with st.spinner("Analyzing..."):
                predictions = predict_resume(resume_text_ml)
                if predictions:
                    st.success("✅ Top Role Predictions:")
                    for i, (label, score) in enumerate(predictions, 1):
                        st.markdown(f"**{i}. {label}** – {score:.2%} confidence")
                    st.session_state["top_role"] = predictions[0][0]

                    try:
                        validated = validate_role_with_gemini(resume_text_ml, [p[0] for p in predictions])
                        st.markdown(f"🔎 **Gemini-Validated Role**: `{validated}`")
                    except Exception as e:
                        st.warning(f"Gemini validation error: {e}")
                else:
                    st.warning("❗ No high-confidence roles predicted.")

# ------------------------ TAB 2 ------------------------
with tab2:
    st.header("💡 Resume Feedback via Gemini AI")
    uploaded_file_gemini = st.file_uploader("📤 Upload your resume (PDF)", type="pdf", key="genai_tab")

    if uploaded_file_gemini:
        resume_text_genai = extract_text_from_pdf(uploaded_file_gemini)
        st.text_area("📄 Extracted Resume Text", resume_text_genai, height=250, key="resume_text_genai_box")

        target_role = st.text_input("🎯 Enter your target job role (e.g. Data Analyst, DevOps Engineer)")

        if st.button("🧠 Get Gemini Feedback"):
            if resume_text_genai and target_role:
                with st.spinner("Generating feedback..."):
                    feedback = get_resume_feedback_gemini(resume_text_genai, target_role)
                    st.subheader("💬 Suggestions")
                    st.info(feedback)
            else:
                st.warning("Please provide both the resume and target job role.")
