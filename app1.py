import os
import pickle
import logging
from flask import Flask, render_template, request, jsonify, abort
import re
import pdfminer.high_level
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample  # Import resample


app = Flask(__name__)
logging.basicConfig(filename='app.log', level=logging.ERROR)

def load_model_and_vectorizer():
    #  Replace these with the ACTUAL full paths on your system
    model_path = r"C:\Users\asus\Downloads\projectibm\projectibm\Resume-Screening-with-Machine-Learning-Job-Recommendations-Parsing-Categorization-main\models\rf_classifier_categorization.pkl"
    vectorizer_path = r"C:\Users\asus\Downloads\projectibm\projectibm\Resume-Screening-with-Machine-Learning-Job-Recommendations-Parsing-Categorization-main\models\tfidf_vectorizer_categorization.pkl"
    rf_classifier = None
    tfidf_vectorizer = None

    try:
        with open(model_path, 'rb') as model_file:
            rf_classifier = pickle.load(model_file)
        with open(vectorizer_path, 'rb') as vectorizer_file:
            tfidf_vectorizer = pickle.load(vectorizer_file)
        logging.info("Model and vectorizer loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Model or vectorizer file not found.")
        return None, None
    except Exception as e:
        logging.error(f"Error loading model/vectorizer: {e}")
        return None, None

    return rf_classifier, tfidf_vectorizer

rf_classifier, tfidf_vectorizer = load_model_and_vectorizer()

if rf_classifier is None or tfidf_vectorizer is None:
    logging.error("Failed to load model files. Application will exit.")
    abort(500, "Failed to load model files. The application cannot process resume uploads.")


def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer.transform([resume_text])
    predicted_category = rf_classifier.predict(resume_tfidf)[0]
    return predicted_category

def extract_name_from_resume(text):
    name_match = re.search(r"Name:\s*([A-Za-z\s]+)", text)
    if name_match:
        return name_match.group(1).strip()
    name_match = re.search(r"^(?:[A-Z][a-z]+\s+){1,2}[A-Z][a-z]+", text, re.MULTILINE)
    if name_match:
        return name_match.group().strip()
    return None

def extract_contact_number_from_resume(text):
    phone_match = re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    if phone_match:
        return phone_match.group()
    return None

def extract_email_from_resume(text):
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}", text)
    if email_match:
        return email_match.group()
    return None

def extract_skills_from_resume(text):
    skills_keywords = [
        "Python", "Java", "SQL", "Javascript", "HTML", "CSS", "Machine Learning", "Deep Learning",
        "Figma", "UI/UX", "Teamwork", "Communication", "Leadership", "Time Management"
    ]
    extracted_skills = []
    for skill in skills_keywords:
        if re.search(r"\b" + re.escape(skill) + r"\b", text, re.IGNORECASE):
            extracted_skills.append(skill)
    return extracted_skills

def extract_education_from_resume(text):
    education = []
    education_keywords = [
        'Computer Science', 'Information Technology', 'Software Engineering', 'Electrical Engineering',
        'Mechanical Engineering', 'Civil Engineering',
        'Chemical Engineering', 'Biomedical Engineering', 'Aerospace Engineering', 'Nuclear Engineering',
        'Industrial Engineering', 'Systems Engineering',
        'Environmental Engineering', 'Petroleum Engineering', 'Geological Engineering', 'Marine Engineering',
        'Robotics Engineering', 'Biotechnology',
        'Biochemistry', 'Microbiology', 'Genetics', 'Molecular Biology', 'Bioinformatics', 'Neuroscience',
        'Biophysics', 'Biostatistics', 'Pharmacology',
        'Physiology', 'Anatomy', 'Pathology', 'Immunology', 'Epidemiology', 'Public Health',
        'Health Administration', 'Nursing', 'Medicine', 'Dentistry',
        'Pharmacy', 'Veterinary Medicine', 'Medical Technology', 'Radiography', 'Physical Therapy',
        'Occupational Therapy', 'Speech Therapy', 'Nutrition',
        'Sports Science', 'Kinesiology', 'Exercise Physiology', 'Sports Medicine', 'Rehabilitation Science',
        'Psychology', 'Counseling', 'Social Work',
        'Sociology', 'Anthropology', 'Criminal Justice', 'Political Science', 'International Relations',
        'Economics', 'Finance', 'Accounting', 'Business Administration',
        'Management', 'Marketing', 'Entrepreneurship', 'Hospitality Management', 'Tourism Management',
        'Supply Chain Management', 'Logistics Management',
        'Operations Management', 'Human Resource Management', 'Organizational Behavior',
        'Project Management', 'Quality Management', 'Risk Management',
        'Strategic Management', 'Public Administration', 'Urban Planning', 'Architecture', 'Interior Design',
        'Landscape Architecture', 'Fine Arts',
        'Visual Arts', 'Graphic Design', 'Fashion Design', 'Industrial Design', 'Product Design',
        'Animation', 'Film Studies', 'Media Studies',
        'Communication Studies', 'Journalism', 'Broadcasting', 'Creative Writing', 'English Literature',
        'Linguistics', 'Translation Studies',
        'Foreign Languages', 'Modern Languages', 'Classical Studies', 'History', 'Archaeology',
        'Philosophy', 'Theology', 'Religious Studies',
        'Ethics', 'Education', 'Early Childhood Education', 'Elementary Education', 'Secondary Education',
        'Special Education', 'Higher Education',
        'Adult Education', 'Distance Education', 'Online Education', 'Instructional Design',
        'Curriculum Development',
        'Library Science', 'Information Science', 'Computer Engineering', 'Software Development',
        'Cybersecurity', 'Information Security',
        'Network Engineering', 'Data Science', 'Data Analytics', 'Business Analytics', 'Operations Research',
        'Decision Sciences',
        'Human-Computer Interaction', 'User Experience Design', 'User Interface Design', 'Digital Marketing',
        'Content Strategy',
        'Brand Management', 'Public Relations', 'Corporate Communications', 'Media Production', 'Digital Media',
        'Web Development',
        'Mobile App Development', 'Game Development', 'Virtual Reality', 'Augmented Reality',
        'Blockchain Technology', 'Cryptocurrency',
        'Digital Forensics', 'Forensic Science', 'Criminalistics', 'Crime Scene Investigation', 'Emergency Management',
        'Fire Science',
        'Environmental Science', 'Climate Science', 'Meteorology', 'Geography', 'Geomatics', 'Remote Sensing',
        'Geoinformatics',
        'Cartography', 'GIS (Geographic Information Systems)', 'Environmental Management', 'Sustainability Studies',
        'Renewable Energy',
        'Green Technology', 'Ecology', 'Conservation Biology', 'Wildlife Biology', 'Zoology'
    ]
    for keyword in education_keywords:
        pattern = r"(?i)\b{}\b".format(re.escape(keyword))
        match = re.search(pattern, text)
        if match:
            education.append(match.group())
    return education

def get_job_recommendation(category):
    """
    This is a placeholder. You need to implement the actual job recommendation logic.
    For example, you might have a dictionary that maps categories to job titles.
    """
    job_recommendations = {
        "HR": "HR Generalist",
        "DESIGNER": "Graphic Designer",
        # ... add more mappings
    }
    return job_recommendations.get(category, "Job recommendations not available for this category")

@app.route('/')
def index():
    return render_template('index.html')  # Assuming your HTML file is named index.html

@app.route('/pred', methods=['POST'])
def predict():
    if 'resume' not in request.files:
        return "No file part"
    file = request.files['resume']
    if file.filename == '':
        return "No selected file"

    if file:
        try:
            file_content = file.read()
            file_extension = file.filename.rsplit('.', 1)[1].lower()

            if file_extension == 'pdf':
                text = pdfminer.high_level.extract_text(file)  # Pass the file object
            elif file_extension == 'txt':
                text = file_content.decode('utf-8')  # Or another encoding if needed
            else:
                return "Invalid file type"

            predicted_category = predict_category(text)
            # Assuming you have a function to get a job recommendation based on the category
            recommended_job = get_job_recommendation(predicted_category)
            name = extract_name_from_resume(text)
            phone = extract_contact_number_from_resume(text)
            email = extract_email_from_resume(text)
            extracted_skills = extract_skills_from_resume(text)
            extracted_education = extract_education_from_resume(text)

            return render_template('index.html',
                                   predicted_category=predicted_category,
                                   recommended_job=recommended_job,
                                   name=name,
                                   phone=phone,
                                   email=email,
                                   extracted_skills=extracted_skills,
                                   extracted_education=extracted_education)

        except Exception as e:
            return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)