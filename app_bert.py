import fitz  # PyMuPDF
import os
import pandas as pd
import streamlit as st
import sqlite3
from sentence_transformers import SentenceTransformer, util
from datetime import datetime

# Skill Dictionary
SKILLS = [
    "python", "sql", "machine learning", "deep learning",
    "tensorflow", "pytorch", "nlp", "computer vision",
    "data analysis", "statistics", "regression",
    "clustering", "tableau", "power bi", "excel", "R",
    "Scikit-learn", "Keras", "Apache Spark", "Hadoop",
    "Kafka", "AWS", "Azure", "GCP", "Pandas", "NumPy",
    "Dask", "Matplotlib", "Seaborn", "Docker", "Kubernetes",
    "Flask", "FastAPI", "MLflow", "Kubeflow", "Airflow",
    "NLTK", "spaCy", "Hugging Face", "OpenCV", "H2O.ai",
    "Auto-sklearn", "Git", "GitHub", "Statsmodels", "Bayesian Methods"
]


#  Load BERT Model
@st.cache_resource
def load_bert():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_bert()


#  Load Resumes from CSV
def load_resumes_from_csv(csv_file):
    try:
        df = pd.read_csv(csv_file, delimiter=',', engine='python', encoding='latin1')
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return {}
    resumes = {}
    for _, row in df.iterrows():
        candidate = row['ID']
        text = row['Resume_str']
        resumes[candidate] = text
    return resumes


#  Load Resumes from PDF
def extract_text_from_pdf(pdf_file):
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error processing PDF file: {e}")
        return None


#  Extract Skills
def extract_skills(text):
    text_lower = text.lower()
    found = [skill.lower() for skill in SKILLS if skill.lower() in text_lower]
    return list(set(found))


#  Compute Hybrid Match Score
def compute_match_score(candidate_text, jd_text, jd_skills):
    # NER score
    cand_skills = extract_skills(candidate_text)
    common = set(cand_skills).intersection(jd_skills)
    ner_score = (len(common) / len(jd_skills) * 100) if jd_skills else 0

    # BERT semantic score
    cand_emb = embedder.encode(candidate_text, convert_to_tensor=True)
    jd_emb = embedder.encode(jd_text, convert_to_tensor=True)
    bert_score = float(util.cos_sim(cand_emb, jd_emb)[0][0]) * 100

    # Default Weights
    w_ner = 0.6
    w_bert = 0.6

    hybrid = w_ner * ner_score + w_bert * bert_score
    shortlist_prob = min(100, hybrid * 1.2)
    missing = list(set(jd_skills) - set(cand_skills))

    return {
        "Match Percentage": round(hybrid, 2),
        "Hybrid Score": round(hybrid, 2),
        "Shortlist Probability": round(shortlist_prob, 2),
        "Missing Key Points": missing
    }


#  Rank Candidates
def rank_candidates(resumes, jd_text):
    jd_skills = extract_skills(jd_text)
    results = []
    for cand, text in resumes.items():
        data = compute_match_score(text, jd_text, jd_skills)
        results.append({"Candidate": cand, **data})
    return sorted(results, key=lambda x: x["Hybrid Score"], reverse=True)


#  Save to SQLite
def save_to_db(results):
    conn = sqlite3.connect("resumes.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS resume_scores (
            timestamp TEXT,
            candidate TEXT,
            hybrid_score REAL,
            match_percentage REAL,
            shortlist_probability REAL,
            missing_skills TEXT
        )
    """)
    for r in results:
        c.execute("INSERT INTO resume_scores VALUES (?,?,?,?,?,?)", (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            r["Candidate"],
            r["Hybrid Score"],
            r["Match Percentage"],
            r["Shortlist Probability"],
            ', '.join(r["Missing Key Points"])
        ))
    conn.commit()
    conn.close()

#  Streamlit UI
def main():
    st.title("🧠 AI-Powered Resume Screener")
    st.write("Upload resumes (CSV or PDF) and paste a job description to screen candidates based on skill and semantic match.")

    files = st.file_uploader("Upload Resume Files (CSV or PDF)", type=["csv", "pdf"], accept_multiple_files=True)
    jd_text = st.text_area("Paste Job Description", height=200)

    if st.button("🔍 Rank Candidates"):
        if not files:
            st.error("Please upload at least one resume file.")
        elif not jd_text.strip():
            st.error("Please enter a job description.")
        else:
            resumes = {}
            for f in files:
                if f.type == "text/csv":
                    resumes.update(load_resumes_from_csv(f))
                else:
                    txt = extract_text_from_pdf(f)
                    if txt:
                        resumes[f.name.replace(".pdf", "")] = txt

            results = rank_candidates(resumes, jd_text)
            save_to_db(results)

            df = pd.DataFrame(results)
            st.success("🎉 Ranking Complete!")
            st.dataframe(df[["Candidate", "Match Percentage", "Shortlist Probability"]])

            for _, row in df.iterrows():
                st.write(f"**{row['Candidate']}** - Missing Skills: {', '.join(row['Missing Key Points']) or 'None'}")

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Report CSV", data=csv, file_name="resume_ranking_report.csv")

if __name__ == "__main__":
    main()
