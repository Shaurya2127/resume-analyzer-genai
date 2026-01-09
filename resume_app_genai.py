import numpy as np
import pickle
import re
import nltk
import fitz
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
# NLTK SETUP
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
# PDF EXTRACTION
# =====================================================
def extract_text_from_pdf(uploaded_file):
    try:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return " ".join(page.get_text() for page in doc).strip()
    except Exception:
        return ""

# =====================================================
# LOAD ML MODELS
# =====================================================
with open("resume_classifier_ml.pkl", "rb") as f:
    ml_model = pickle.load(f)

with open("label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

# =====================================================
# GEMINI – SAFE + RATE LIMITED
# =====================================================
@st.cache_data(show_spinner=False)
def cached_gemini(prompt):
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    models = [
        m.name for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]

    if not models:
        return "⚠️ Gemini models are not available for this API key."

    model = genai.GenerativeModel(models[0])
    return model.generate_content(prompt).text


def gemini_call(prompt):
    try:
        return cached_gemini(prompt)
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            return (
                "⚠️ Gemini rate limit reached.\n"
                "Please wait a few seconds or upgrade your plan."
            )
        return f"⚠️ Gemini error: {e}"

# =====================================================
# GEMINI TASKS (TARGET-ROLE FIRST)
# =====================================================
def get_alignment_feedback(resume_text, target_role):
    prompt = f"""
You are a senior hiring manager.

Analyze how well this resume matches the target role: {target_role}.
Give:
1. Alignment score (High / Medium / Low)
2. Missing skills
3. Key improvements needed

Resume:
{resume_text}
"""
    return gemini_call(prompt)


def get_resume_feedback(resume_text, target_role):
    prompt = f"""
You are a professional resume coach.

Improve this resume for a {target_role} role.
Focus on:
- Structure
- Skill gaps
- Bullet points
- ATS optimization

Resume:
{resume_text}
"""
    return gemini_call(prompt)


def generate_improved_resume(resume_text, target_role):
    prompt = f"""
Rewrite this resume for a {target_role} role.
Use strong action verbs, quantified impact,
and ATS-friendly keywords.

Resume:
{resume_text}
"""
    return gemini_call(prompt)

# =====================================================
# RESUME SCORE (TARGET ROLE BASED)
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
Target-role driven resume analysis  
Built by **Shaurya Chauhan**
""")

st.title("🧠 Resume Analyzer & Optimizer")

uploaded_file = st.file_uploader(
    "📤 Upload Resume (PDF)",
    type="pdf"
)

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)

    if not resume_text:
        st.error("Unable to read resume.")
        st.stop()

    st.text_area(
        "📄 Extracted Resume Text",
        resume_text,
        height=260
    )

    # ============================
    # USER TARGET ROLE
    # ============================
    target_role = st.text_input(
        "🎯 Which job role are you aiming for?",
        placeholder="e.g. Data Scientist, Data Analyst, ML Engineer"
    )

    if not target_role.strip():
        st.info("Please enter your target job role to continue.")
        st.stop()

    # ============================
    # ML ROLE COMPARISON
    # ============================
    cleaned = clean_text(resume_text)
    proba = ml_model.predict_proba([cleaned])[0]
    top_idx = np.argsort(proba)[::-1][:3]

    predictions = [
        (label_encoder.inverse_transform([i])[0], proba[i])
        for i in top_idx
    ]

    st.subheader("📊 ML-Predicted Roles (Reference)")
    for i, (role, score) in enumerate(predictions, 1):
        st.markdown(f"**{i}. {role}** — {score:.2%}")

    top_ml_role, top_conf = predictions[0]

    resume_score = calculate_resume_score(resume_text, top_conf)
    st.metric("📊 Resume Strength Score", f"{resume_score} / 100")

    # ============================
    # GEMINI ANALYSIS
    # ============================
    st.subheader("🤖 AI Analysis for Your Target Role")

    if st.button("🔎 Analyze Alignment"):
        alignment = get_alignment_feedback(resume_text, target_role)
        st.info(alignment)

    if st.button("🧠 Get Resume Improvement Suggestions"):
        feedback = get_resume_feedback(resume_text, target_role)
        st.info(feedback)

    if st.button("✨ Generate Target-Role Optimized Resume"):
        improved = generate_improved_resume(resume_text, target_role)

        st.text_area(
            "📄 Improved Resume (Target Role Optimized)",
            improved,
            height=350
        )

        st.download_button(
            "📥 Download Improved Resume",
            improved,
            file_name=f"Improved_Resume_{target_role.replace(' ', '_')}.txt",
            mime="text/plain"
        )
