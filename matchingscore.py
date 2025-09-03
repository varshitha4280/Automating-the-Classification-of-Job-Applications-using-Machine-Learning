import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

def preprocess_text(text):
    """Cleans and tokenizes text for analysis."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def calculate_match_score(job_description, resume):
    """Calculates a match score between a job description and a resume."""

    job_tokens = preprocess_text(job_description)
    resume_tokens = preprocess_text(resume)

    job_keywords = Counter(job_tokens)
    resume_keywords = Counter(resume_tokens)

    # Calculate the intersection of keywords
    common_keywords = set(job_keywords.keys()) & set(resume_keywords.keys())

    # Calculate a score based on the frequency of common keywords
    score = sum(min(job_keywords[keyword], resume_keywords[keyword]) for keyword in common_keywords)

    # Normalize the score (optional)
    # You can normalize by the total number of job description keywords
    if len(job_keywords) > 0:
        score /= len(job_keywords)

    return score

# Example usage
job_description = """
Software Engineer , Python, Django, and REST APIs, 3+ years of experience, Bachelor's degree in Computer Science.
"""

resume = """
I am a Software Engineer with 5 years of experience in Python and Django. 
I have built several REST APIs and hold a Master's degree in Computer Science.
"""

match_score = calculate_match_score(job_description, resume)
print(f"Match Score: {match_score}")