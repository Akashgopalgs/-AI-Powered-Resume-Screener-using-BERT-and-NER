# app1.py
import streamlit as st
import pandas as pd
from resume_screener import (
    load_resumes_from_csv, extract_text_from_pdf, rank_candidates, save_to_db
)

def main():
    st.title("🧠 AI-Powered Resume Screener")
    st.write("This app ranks candidates using BERT and NER by matching resumes to job descriptions..")

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
