# 🧠 Resume Analyzer + GenAI Optimizer

A Streamlit app that predicts the best job role from a resume using **Machine Learning**, and enhances it using **Gemini Pro** by Google.

## 🔍 Features

- 📤 Upload your resume in PDF
- 🧠 Get top job role prediction using a trained Random Forest model
- 🤖 Get resume improvement tips using Gemini 1.5 Pro
- 📊 Confidence scores per predicted role
- 💬 Interactive UI built with Streamlit

---

## 📂 Project Structure
resume-analyzer-genai/ <br/>
├── resume_app_genai.py # Streamlit app <br/>
├── resume_classifier_ml.pkl # ML model <br/>
├── label_encoder.pickle # Encoded role labels <br/>
├── requirements.txt # Python dependencies <br/>
├── .streamlit/ <br/>
 └── secrets.toml # API key (private) <br/>
├── README.md # This file <br/>


---

## 🔧 Setup Locally

### 1. Clone the repo
```bash
git clone https://github.com/Shaurya2127/resume-analyzer-genai.git
cd resume-analyzer-genai
```
### 2. Create .streamlit/secrets.toml
GEMINI_API_KEY = "your-gemini-api-key-here"
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the app
```bash
streamlit run resume_app_genai.py
```

🚀 Live Demo
🔗 Streamlit App (Live) - [ https://resume-analyzer-genai-kuqpbfhz9wx7fhj8i6oz77.streamlit.app/ ]

📚 Technologies Used
Python, Scikit-learn, Pandas, NLTK

Google Gemini API (google-generativeai)

Streamlit (frontend)

PyMuPDF (for PDF text extraction)

📧 shauryachauhan721@gmail.com
🔗 LinkedIn - [   
🔗 GitHub
