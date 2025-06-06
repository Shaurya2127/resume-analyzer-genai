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

# 🧠 GPT Feedback Function
import google.generativeai as genai
import os
api_key = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=api_key)
print("Gemini API key loaded:", api_key)
genai.configure(api_key=api_key)
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

# 🖥 Streamlit Configuration
st.set_page_config(page_title="Resume Optimizer", page_icon="🧠", layout="wide")

# Sidebar Branding
st.sidebar.title("📄 Resume Optimizer")
st.sidebar.markdown("""
**Powered by GenAI + ML**
Upload your resume and receive:
- 🔍 Role Predictions
- 💡 GPT Feedback
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Shaurya Chauhan**")
st.sidebar.markdown("[GitHub](https://github.com/Shaurya2127) | [LinkedIn](https://www.linkedin.com/in/shaurya-chauhan-0089911bb/)")

# Header
st.title("🧠 Resume Analyzer & Optimizer")
st.markdown("<h5 style='color: gray;'>Get job-fit predictions and GPT-based resume suggestions</h5>", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns(2)

resume_text = ""

with col1:
    uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])
    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)

with col2:
    if resume_text:
        st.text_area("📋 Extracted Resume Text", resume_text, height=300)

# Prediction and GPT Tabs
if resume_text:
    tab1, tab2 = st.tabs(["📊 ML Role Prediction", "💡 GPT Resume Feedback"])

    with tab1:
        def predict_resume(text):
            cleaned = clean_text(text)
            proba = model.predict_proba([cleaned])[0]
            top_idx = np.argsort(proba)[::-1][:3]
            top_labels = [(label_encoder.inverse_transform([i])[0], round(proba[i]*100, 2)) for i in top_indices]
            top_scores = proba[top_idx]
            return list(zip(top_labels, top_scores))

        if st.button("🔍 Predict Job Role"):
            with st.spinner("Analyzing with ML model..."):
                predictions = predict_resume(resume_text)
                st.success("✅ Top Predictions:")
                print("Extracted resume text:\n", resume_text)
                for i, (label, score) in enumerate(predictions, 1):
                    st.markdown(f"**{i}. {label}** – {score:.2%} confidence")
            st.session_state["top_role"] = predictions[0][0]

    with tab2:
        if st.button("🧠 Get GPT Feedback"):
            with st.spinner("Contacting GPT..."):
                top_role = st.session_state.get("top_role", "Data Analyst")
                feedback = get_resume_feedback_gemini(resume_text, top_role)
                st.markdown("### 💬 GPT Suggestions")
                st.info(feedback)

st.markdown("---")
st.markdown("Made by **Shaurya Chauhan**")
