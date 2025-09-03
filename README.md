# Automating-the-Classification-of-Job-Applications-using-Machine-Learning

🤖 Automating the Classification of Job Applications using Machine Learning
📌 Overview

This project focuses on automating the resume screening process using Machine Learning and Natural Language Processing (NLP). Recruiters face challenges in handling large volumes of applications, which makes manual screening slow, costly, and prone to bias. Our solution provides an intelligent system that can classify resumes, analyze candidate-job fit, and recommend suitable opportunities, improving recruitment efficiency.

🚀 Features

📂 Resume Parsing – Extracts key details (name, skills, education, experience) from resumes (PDF/TXT).

🏷️ Resume Categorization – Classifies resumes into predefined job categories using ML models.

🔍 Job Match Scoring – Calculates similarity between candidate resumes and job descriptions.

🌐 Flask Web App – Upload resumes, enter job descriptions, and view classification & match results.

📊 Accuracy – Achieved ~84% classification accuracy, with some categories reaching 100% precision and recall.

⚖️ Bias Reduction – Promotes fair hiring by minimizing unconscious bias through automated evaluation.

🛠️ Tech Stack

Programming Language: Python

Frameworks: Flask (backend web framework)

Libraries: NLP (NLTK, spaCy), Scikit-learn, PDFMiner, Regex

Frontend: HTML, CSS, JavaScript

Deployment: Flask API + Web Interface

🎯 Objectives

Streamline recruitment by reducing time and cost in manual screening.

Provide personalized job recommendations based on candidate profiles.

Improve accuracy and fairness in candidate evaluation.

Build a scalable web-based tool accessible to recruiters and job seekers.

📂 System Workflow

Upload Resume → Extract text (PDFMiner, NLP preprocessing).

Text Processing → Clean text, remove stopwords, apply lemmatization.

Resume Categorization → ML model predicts job category.

Job Description Matching → Compute similarity score with job descriptions.

Results → Display job fit score + recommended categories in the web UI.

📊 Results

✅ Achieved 84% classification accuracy.

✅ Strong performance in domains like Automobile and BPO (perfect precision & recall).

✅ Improved recruitment efficiency by reducing screening time.

📸 Screenshots (Optional to add)

Resume upload page

Job description input page

Candidate classification & match score results

📌 Future Enhancements

Integration with cloud-based databases for scalability.

Support for multiple resume formats (DOCX, LinkedIn exports).

Advanced deep learning models (BERT, transformers) for higher accuracy.

Recruitment analytics dashboard for insights and trends.

🏆 Conclusion

This project successfully demonstrates how AI and NLP can revolutionize recruitment by automating resume classification and job matching. It reduces bias, saves time, and improves hiring outcomes, offering a scalable solution for SMBs, enterprises, and recruitment agencies alike.
