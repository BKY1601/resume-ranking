import streamlit as st
from PyPDF2 import PdfReader
import docx2txt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os

# Background CSS
page_bg_img = '''
<style>
    .stApp {
        background-image: url("https://github.com/BKY1601/resume-ranking/blob/main/res/img/bg.jpg?raw=true");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .title {
        font-family: 'Exo 2', sans-serif;
        font-style: italic;
        font-weight: bold;
        color: white;
        text-align: center;
        font-size: 36px;
        padding: 20px;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.7);
    }
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown('<h1 class="title">AI Resume Ranker</h1>', unsafe_allow_html=True)

# Extract text from resumes
def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(file)
        return "".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif ext == ".docx":
        return docx2txt.process(file)
    else:
        return ""

# Rank resumes using cosine similarity
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    vectors = tfidf_matrix.toarray()
    job_vec = vectors[0]
    resume_vecs = vectors[1:]
    similarities = cosine_similarity([job_vec], resume_vecs).flatten()
    return similarities, tfidf_vectorizer, tfidf_matrix

# Input: Job description
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# Input: Resumes
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")
    resumes = []
    filenames = []
    for file in uploaded_files:
        resumes.append(extract_text(file))
        filenames.append(file.name)

    scores, vectorizer, all_vectors = rank_resumes(job_description, resumes)
    percentages = [round(score * 100, 2) for score in scores]

    df = pd.DataFrame({"Resume": filenames, "Match (%)": percentages})
    df = df.sort_values(by="Match (%)", ascending=False).reset_index(drop=True)
    st.write(df)

    # Ranking graph
    if st.checkbox("Show Ranking Graph"):
        st.subheader("Resume Ranking Graph")
        fig, ax = plt.subplots(figsize=(8, len(filenames) * 0.5))
        bars = ax.barh(df["Resume"], df["Match (%)"], color="skyblue", height=0.3)
        ax.set_xlabel("Match %")
        ax.invert_yaxis()
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height() / 2, f"{width:.2f}%", va='center')
        st.pyplot(fig)

    # Keyword match display
    if st.checkbox("Show matched Skills & requirements"):
        st.subheader("Matched requirements")
        keywords = vectorizer.get_feature_names_out()
        jd_vec = all_vectors[0].toarray().flatten()
        jd_keywords = set([keywords[i] for i, val in enumerate(jd_vec) if val > 0])

        for i in range(1, all_vectors.shape[0]):
            resume_vec = all_vectors[i].toarray().flatten()
            resume_keywords = set([keywords[j] for j, val in enumerate(resume_vec) if val > 0])
            matched = sorted(jd_keywords.intersection(resume_keywords))
            st.markdown(f"**{filenames[i-1]}**")
            if matched:
                st.markdown("✅ Matched requirements:")
                st.markdown(
                    " ".join([
                        f"<span style='background-color:#D3F8E2; padding:4px; border-radius:5px;'>{kw}</span>"
                        for kw in matched
                    ]),
                    unsafe_allow_html=True
                )
            else:
                st.markdown("⚠️ No relevant keywords matched.")
