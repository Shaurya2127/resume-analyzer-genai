import numpy as np
import pickle
import re
import time
import nltk
import fitz  # PyMuPDF
import streamlit as st
import google.generativeai as genai

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(
    page_title="Resume Analyzer & Optimizer",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# NLTK SETUP (CLOUD SAFE)
# =====================================================
@st.cache_resource
def setup_nltk():
    nltk.download("stopwords")
    nltk.download("wordnet")

setup_nltk()
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# =====================================================
# TEXT CLEANING
# =====================================================
def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower().split()
    return " ".join(
        lemmatizer.lemmatize(w) for w in text if w not in stop_words
    )

# =====================================================
# PDF TEXT EXTRACTION
# =====================================================
def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return " ".join(page.get_text() for page in doc).strip()
    except Exception:
        return ""

# =====================================================
# LOAD ML MODEL
# =====================================================
with open("resume_classifier_ml.pkl", "rb") as f:
    ml_model = pickle.load(f)

with open("label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

# =====================================================
# GEMINI – RATE-LIMIT SAFE & FAIL-PROOF
# =====================================================
@st.cache_data(show_spinner=False)
def cached_gemini_response(prompt):
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    models = [
        m.name for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]

    if not models:
        return "⚠️ Gemini is enabled but no text-generation models are available for this API key."

    model = genai.GenerativeModel(models[0])
    response = model.generate_content(prompt)
    return response.text


def gemini_call(prompt):
    try:
        return cached_gemini_response(prompt)

    except Exception as e:
        msg = str(e).lower()

        if "429" in msg or "quota" in msg:
            return (
                "⚠️ Gemini free-tier rate limit reached.\n\n"
                "Please wait ~15 seconds and try again, "
                "or upgrade your Gemini plan."
            )

        return f"⚠️ Gemini error: {e}"

# =====================================================
# GEMINI FUNCTIONS
# =====================================================
def validate_role_with_gemini(resume_text, roles):
    prompt = f"""
You are an AI career assistant.
Choose the best matching role from the list below.

Roles: {', '.join(roles)}

Resume:
{resume_text}
"""
    return gemini_call(prompt)


def get_resume_feedback(resume_text, role):
    prompt = f"""
You are a professional resume coach.
Give actionable feedback for a {role} role under:
- Structure
- Skills gaps
- Bullet point improvements
- ATS optimization

Resume:
{resume_text}
"""
    return gemini_call(prompt)


def generate_improved_resume(resume_text, role):
    prompt = f"""
Rewrite this resume for a {role} role.
Improve clarity, bullet points, and ATS keywords.
Keep it professional.

Resume:
{resume_text}
"""
    return gemini_call(prompt)

# =====================================================
# RESUME SCORE
# =====================================================
def calculate_resume_score(resume_text, confidence):
    wc = len(resume_text.split())
    length_score = 1.0 if 300 <= wc <= 900 else 0.6
    return round(min((0.6 * confidence + 0.4 * length_score) * 100, 100), 1)

# =====================================================
# UI
# =====================================================
st.sidebar.title("📄 Resume Optimizer")
st.sidebar.markdown("""
ML + GenAI powered resume analysis  
Built by **Shaurya Chauhan**
""")

st.title("🧠 Resume Analyzer & Optimizer")

uploaded_file = st.file_uploader(
    "📤 Upload Resume (PDF)",
    type="pdf",
    key="resume_uploader"
)

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)

    if not resume_text:
        st.error("Unable to read PDF.")
        st.stop()

    st.text_area(
        "📄 Extracted Resume Text",
        resume_text,
        height=260,
        key="resume_text_area"
    )

    # ============================
    # ML PREDICTION
    # ============================
    cleaned = clean_text(resume_text)
    proba = ml_model.predict_proba([cleaned])[0]
    top_idx = np.argsort(proba)[::-1][:3]

    predictions = [
        (label_encoder.inverse_transform([i])[0], proba[i])
        for i in top_idx
    ]

    st.subheader("📊 ML Role Predictions")
    for i, (role, score) in enumerate(predictions, 1):
        st.markdown(f"**{i}. {role}** — {score:.2%}")

    top_role, top_conf = predictions[0]
    resume_score = calculate_resume_score(resume_text, top_conf)

    st.metric("📊 Resume Strength Score", f"{resume_score} / 100")

    # ============================
    # GEMINI SECTION
    # ============================
    st.subheader("🤖 Gemini AI Insights")

    if st.button("🔎 Validate Role with Gemini"):
        validated = validate_role_with_gemini(
            resume_text, [r for r, _ in predictions]
        )
        st.info(validated)

    if st.button("🧠 Get AI Resume Feedback"):
        feedback = get_resume_feedback(resume_text, top_role)
        st.info(feedback)

    if st.button("✨ Generate Improved Resume"):
        improved = generate_improved_resume(resume_text, top_role)

        st.text_area(
            "📄 Improved Resume (AI Optimized)",
            improved,
            height=350,
            key="improved_resume_area"
        )

        st.download_button(
            "📥 Download Improved Resume",
            improved,
            file_name="Improved_Resume.txt",
            mime="text/plain",
            key="download_btn"
        )
