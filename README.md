# 🧠 AI Resume Ranker

AI Resume Ranker is a Python project that compares and ranks resumes based on a given job description using **TF-IDF** and **cosine similarity**. It helps recruiters find the most relevant resumes automatically using Natural Language Processing (NLP).

---

## 🚀 Features

- Read multiple resumes in PDF/DOCX format
- Extract job description from text
- Convert both to TF-IDF vectors
- Calculate similarity using cosine similarity
- Rank resumes by relevance

---

## 🛠 Tech Stack

- Python  
- Scikit-learn  
- Pandas  
- NumPy  
- PyPDF2 / python-docx
- matplotlib
- Pandas

---

## ▶️ How to Run

1. Clone the repo
2. Setup anconda streamlit
3. Install required packages
4. Run the main script (`resume_ranking.py`)
5. Paste job description in job_description section
6. Upload the resumes (pdf/docx format only)

---

## 📊 Sample Output

A table will be generated can be imported in csv format:

| Resume       | Score    |
|--------------|----------|
| resume1.pdf  | 82.1%    |
| resume2.docx | 75.3%    |

---

## 👨‍💻 Author

**Bipin Yadav**  
📧 bipinyadav919@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/bipin-yadav-jan16)  
🔗 [GitHub](https://github.com/BKY1601)                                                                                                   
🔗 [Live project Link](https://resume-ranking-by-bky.streamlit.app/)
