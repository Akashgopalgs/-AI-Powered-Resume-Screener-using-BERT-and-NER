
import fitz  # PyMuPDF
import pandas as pd
import sqlite3
import pickle
from datetime import datetime
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')


# Predefined skills
SKILLS = [
    "python", "sql", "machine learning", "deep learning", "tensorflow", "pytorch", "nlp",
    "computer vision", "data analysis", "statistics", "regression", "clustering", "tableau",
    "power bi", "excel", "R", "Scikit-learn", "Keras", "Apache Spark", "Hadoop", "Kafka",
    "AWS", "Azure", "GCP", "Pandas", "NumPy", "Dask", "Matplotlib", "Seaborn", "Docker",
    "Kubernetes", "Flask", "FastAPI", "MLflow", "Kubeflow", "Airflow", "NLTK", "spaCy",
    "Hugging Face", "OpenCV", "H2O.ai", "Auto-sklearn", "Git", "GitHub", "Statsmodels", "Bayesian Methods"
]

def load_resumes_from_csv(csv_file):
    df = pd.read_csv(csv_file, delimiter=',', engine='python', encoding='latin1')
    return {row['ID']: row['Resume_str'] for _, row in df.iterrows()}

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = "\n".join([page.get_text("text") for page in doc])
    return text.strip()

def extract_skills(text):
    text_lower = text.lower()
    return list(set([skill.lower() for skill in SKILLS if skill.lower() in text_lower]))

def compute_match_score(candidate_text, jd_text, jd_skills):
    cand_skills = extract_skills(candidate_text)
    common = set(cand_skills).intersection(jd_skills)
    ner_score = (len(common) / len(jd_skills) * 100) if jd_skills else 0

    cand_emb = embedder.encode(candidate_text, convert_to_tensor=True)
    jd_emb = embedder.encode(jd_text, convert_to_tensor=True)
    bert_score = float(util.cos_sim(cand_emb, jd_emb)[0][0]) * 100

    w_ner = 0.6
    w_bert = 0.4
    hybrid = w_ner * ner_score + w_bert * bert_score
    shortlist_prob = min(100, hybrid * 1.1)
    missing = list(set(jd_skills) - set(cand_skills))

    return {
        "Match Percentage": round(hybrid, 2),
        "Hybrid Score": round(hybrid, 2),
        "Shortlist Probability": round(shortlist_prob, 2),
        "Missing Key Points": missing
    }

def rank_candidates(resumes, jd_text):
    jd_skills = extract_skills(jd_text)
    results = []
    for cand, text in resumes.items():
        data = compute_match_score(text, jd_text, jd_skills)
        results.append({"Candidate": cand, **data})
    return sorted(results, key=lambda x: x["Hybrid Score"], reverse=True)

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
