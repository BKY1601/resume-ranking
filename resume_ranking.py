import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Apply background image using custom CSS
page_bg_img = '''
<style>
    .stApp {
        background-image: url("https://img.freepik.com/free-photo/top-view-desk-concept-with-copy-space_23-2148236823.jpg?t=st=1742299973~exp=1742303573~hmac=46096f78c68df458f5beeb601693a24bc4f15d952213d7697420d4c987887eb1&w=1380");
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
