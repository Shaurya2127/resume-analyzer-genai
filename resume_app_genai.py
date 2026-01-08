import numpy as np
import pickle
import re
import nltk
import fitz  # PyMuPDF
import streamlit as st
import google.generativeai as genai

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# 🔐 SAFE NLTK SETUP
# =========================
@st.cache_resource
def load_nltk():
    nltk.download("stopwords")
    nltk.download("wordnet")

load_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# =========================
# 🧹 TEXT CLEANING
# =========================
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    return " ".join(
        lemmatizer.lemmatize(word) for word in text if word not in stop_words
    )

# =========================
# 📄 PDF EXTRACTION (SAFE)
# =========================
def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = " ".join(page.get_text() for page in doc)
        return text.strip()
    except Exception:
        return ""

# =========================
# 🤖 GEMINI CONFIG
# =========================
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

def validate_role_with_gemini(resume_text, role_options):
    try:
        prompt = f"""
You are an AI career assistant.
Which role best matches this resume?

Options: {', '.join(role_options)}

Resume:
{resume_text}
"""
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(prompt).text.strip()
    except Exception:
        return "Gemini validation unavailable"

def get_resume_feedback_gemini(resume_text, target_role):
    try:
        prompt = f"""
You are a resume coach.
Analyze the resume for a '{target_role}' role.
Suggest improvements in:
- Structure
- Skills gaps
- Bullet points
- ATS optimization

Resume:
{resume_text}
"""
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(prompt).text
    except Exception:
        return "⚠️ AI feedback temporarily unavailable."

def generate_improved_resume(resume_text, target_role):
    try:
        prompt = f"""
Rewrite the resume for a '{target_role}' role.
Improve clarity, bullet points, and ATS keywords.
Keep it professional.

Resume:
{resume_text}
"""
        model = genai.GenerativeModel("gemini-1.5-flash")
        return model.generate_content(prompt).text
    except Exception:
        return "⚠️ Resume generation unavailable."

# =========================
# 📊 RESUME SCORE
# =========================
def calculate_resume_score(resume_text, ml_confidence):
    word_count = len(resume_text.split())

    if word_count < 300:
        length_score = 0.4
    elif word_count <= 900:
        length_score = 1.0
    else:
        length_score = 0.7

    score = (0.6 * ml_confidence + 0.4 * length_score) * 100
    return min(round(score, 1), 100.0)

# =========================
# 🧠 LOAD ML MODELS
# =========================
with open("resume_classifier_ml.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

if not hasattr(model, "predict_proba"):
    st.error("ML model does not support probability prediction.")
    st.stop()

# =========================
# 🖥 STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Resume Analyzer & Optimizer",
    page_icon="🧠",
    layout="wide"
)

st.sidebar.title("📄 Resume Optimizer")
st.sidebar.markdown("""
Upload your resume to:
- 🔍 Predict job roles
- 📊 Get a resume score
- 💡 Receive AI feedback
- 📥 Download improved resume

Built by **Shaurya Chauhan**
[GitHub](https://github.com/Shaurya2127) | [LinkedIn](https://www.linkedin.com/in/shaurya-chauhan-0089911bb/)
""")

st.title("🧠 Resume Analyzer & Optimizer")
st.caption("Production-ready ML + GenAI resume evaluation")

tab1, tab2 = st.tabs(["📊 ML Role Prediction", "💡 Gemini Resume Feedback"])

# =========================
# 📊 TAB 1 — ML
# =========================
with tab1:
    uploaded_file = st.file_uploader("📤 Upload Resume (PDF)", type="pdf",key="resume_uploader")

    if uploaded_file:
        resume_text_ml = extract_text_from_pdf(uploaded_file)

        if not resume_text_ml:
            st.error("Unable to read PDF. Please upload a valid resume.")
            st.stop()

        st.text_area(
           "📄 Extracted Resume Text",
           resume_text_genai,
           height=250,
           key="extracted_resume_text"
        )


        def predict_resume(text):
            cleaned = clean_text(text)
            proba = model.predict_proba([cleaned])[0]
            top_idx = np.argsort(proba)[::-1][:3]
            return [
                (label_encoder.inverse_transform([i])[0], proba[i])
                for i in top_idx if proba[i] >= 0.15
            ]

        if st.button("🔍 Predict Job Role", key="predict_btn"):
            predictions = predict_resume(resume_text_ml)

            if predictions:
                for i, (role, score) in enumerate(predictions, 1):
                    st.markdown(f"**{i}. {role}** — {score:.2%}")

                top_role, top_confidence = predictions[0]
                st.session_state["top_role"] = top_role

                resume_score = calculate_resume_score(resume_text_ml, top_confidence)
                st.metric("📊 Resume Strength Score", f"{resume_score} / 100")

                validated = validate_role_with_gemini(
                    resume_text_ml, [r for r, _ in predictions]
                )
                st.markdown(f"🔎 **Gemini-Validated Role:** `{validated}`")
            else:
                st.warning("No high-confidence roles detected.")

# =========================
# 💡 TAB 2 — GEMINI
# =========================
with tab2:
    uploaded_file_gemini = st.file_uploader(
        "📤 Upload Resume (PDF)", type="pdf", key="genai"
    )

    if uploaded_file_gemini:
        resume_text_genai = extract_text_from_pdf(uploaded_file_gemini)

        if not resume_text_genai:
            st.error("Unable to read PDF.")
            st.stop()

        st.text_area("📄 Extracted Resume Text", resume_text_genai, height=250)

        target_role = st.text_input(
            "🎯 Target Job Role",
            value=st.session_state.get("top_role", "")
        )

        if st.button("🧠 Get Gemini Feedback", key="feedback_btn"):
            if not target_role.strip():
                st.warning("Please enter a target job role.")
            else:
                feedback = get_resume_feedback_gemini(
                    resume_text_genai, target_role
                )
                st.info(feedback)

        if st.button("✨ Generate Improved Resume", key="improve_btn"):
            if not target_role.strip():
                st.warning("Please enter a target job role.")
            else:
                improved_resume = generate_improved_resume(
                    resume_text_genai, target_role
                )

                st.text_area(
                    "📄 Improved Resume (AI Optimized)",
                    improved_resume,
                    height=350
                )

                st.download_button(
                    "📥 Download Improved Resume",
                    improved_resume,
                    file_name="Improved_Resume.txt",
                    mime="text/plain"
                )

