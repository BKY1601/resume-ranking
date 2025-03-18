import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Apply background image using custom CSS
page_bg_img = '''
<style>
    .stApp {
        background-image: url("https://downloadscdn6.freepik.com/23/2148237/2148236823.jpg?filename=top-view-desk-concept-with-copy-space.jpg&token=exp=1742306210~hmac=9260558c15e10ff83a59473dd85b4740&filename=2148236823.jpg&_gl=1*dg6xqf*_gcl_au*OTY2NzkwMzY3LjE3NDIyOTk5MDQ.*_ga*MTk2Mjg0MjMzNC4xNzQyMjk5OTA1*_ga_QWX66025LC*MTc0MjMwMjcwMS4yLjEuMTc0MjMwNTMxMC42MC4wLjA.");
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

# Custom title with Exo 2 font styling
st.markdown('<h1 class="title">AI Resume Ranker</h1>', unsafe_allow_html=True)

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")
    
    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Display scores
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)
    
    st.write(results)
