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
# LOAD ML MODEL (REFERENCE ONLY)
# =====================================================
with open("resume_classifier_ml.pkl", "rb") as f:
    ml_model = pickle.load(f)

with open("label_encoder.pickle", "rb") as f:
    label_encoder = pickle.load(f)

# =====================================================
# ROLE → SKILL MAP (SCORING BASIS)
# =====================================================
ROLE_SKILLS = {
    "data scientist": [
        "python", "sql", "machine learning", "statistics",
        "pandas", "numpy", "scikit", "visualization"
    ],
    "data analyst": [
        "sql", "excel", "power bi", "tableau",
        "dashboards", "analysis", "visualization"
    ],
    "ml engineer": [
        "python", "machine learning", "deep learning",
        "tensorflow", "pytorch", "deployment", "mlops"
    ],
    "software engineer": [
        "python", "java", "c++", "data structures",
        "algorithms", "system design", "api"
    ]
}

# =====================================================
# RESUME STRENGTH SCORE (TARGET ROLE BASED)
# =====================================================
def calculate_skill_match(resume_text, target_role):
    skills = ROLE_SKILLS.get(target_role.lower(), [])
    if not skills:
        return 0.5  # neutral if role not predefined

    resume_text = resume_text.lower()
    matched = sum(1 for skill in skills if skill in resume_text)
    return matched / len(skills)


def calculate_structure_score(resume_text):
    wc = len(resume_text.split())
    if wc < 300:
        return 0.4
    elif wc <= 900:
        return 1.0
    else:
        return 0.7


def calculate_resume_strength(resume_text, target_role):
    skill_score = calculate_skill_match(resume_text, target_role)
    structure_score = calculate_structure_score(resume_text)

    final_score = (
        0.7 * skill_score +
        0.3 * structure_score
    ) * 100

    return round(final_score, 1)

# =====================================================
# GEMINI (RATE-LIMIT SAFE)
# =====================================================
@st.cache_data(show_spinner=False)
def cached_gemini(prompt):
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    models = [
        m.name for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]

    if not models:
        return "⚠️ Gemini models unavailable for this API key."

    model = genai.GenerativeModel(models[0])
    return model.generate_content(prompt).text


def gemini_call(prompt):
    try:
        return cached_gemini(prompt)
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            return (
                "⚠️ Gemini rate limit reached. "
                "Please wait a few seconds or upgrade your plan."
            )
        return f"⚠️ Gemini error: {e}"

# =====================================================
# GEMINI TASKS (TARGET ROLE DRIVEN)
# =====================================================
def get_alignment_feedback(resume_text, target_role):
    prompt = f"""
Analyze how well this resume aligns with the target role: {target_role}.
Provide:
1. Alignment level (High / Medium / Low)
2. Missing skills
3. Key improvements

Resume:
{resume_text}
"""
    return gemini_call(prompt)


def get_resume_feedback(resume_text, target_role):
    prompt = f"""
You are a resume coach.
Improve this resume specifically for a {target_role} role.
Focus on skills, bullet points, and ATS optimization.

Resume:
{resume_text}
"""
    return gemini_call(prompt)


def generate_improved_resume(resume_text, target_role):
    prompt = f"""
Rewrite this resume for a {target_role} role.
Use strong action verbs and ATS-friendly keywords.

Resume:
{resume_text}
"""
    return gemini_call(prompt)

# =====================================================
# UI
# =====================================================
st.sidebar.title("📄 Resume Optimizer")
st.sidebar.markdown("Target-role driven resume analysis")

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

    target_role = st.text_input(
        "🎯 Which job role are you aiming for?",
        placeholder="e.g. Data Scientist, Data Analyst"
    )

    if not target_role.strip():
        st.info("Please enter your target job role.")
        st.stop()

    # ============================
    # ML ROLE REFERENCE
    # ============================
    cleaned = clean_text(resume_text)
    proba = ml_model.predict_proba([cleaned])[0]
    top_idx = np.argsort(proba)[::-1][:3]

    predictions = [
        (label_encoder.inverse_transform([i])[0], proba[i])
        for i in top_idx
    ]

    st.subheader("📊 ML-Predicted Roles (Reference Only)")
    for i, (role, score) in enumerate(predictions, 1):
        st.markdown(f"**{i}. {role}** — {score:.2%}")

    # ============================
    # RESUME STRENGTH SCORE
    # ============================
    strength_score = calculate_resume_strength(resume_text, target_role)
    st.metric("📊 Resume Strength Score (Target Role)", f"{strength_score} / 100")

    # ============================
    # GEMINI ANALYSIS
    # ============================
    st.subheader("🤖 AI Analysis for Target Role")

    if st.button("🔎 Analyze Alignment"):
        alignment = get_alignment_feedback(resume_text, target_role)
        st.info(alignment)

    if st.button("🧠 Get Improvement Suggestions"):
        feedback = get_resume_feedback(resume_text, target_role)
        st.info(feedback)

    if st.button("✨ Generate Optimized Resume"):
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
