from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import re
import string
import emoji
import spacy
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Load the trained model and CountVectorizer
model = joblib.load('best.pkl')
vectorizer = joblib.load('vec.pkl')

# Load the preprocessing resources
nlp = spacy.load('en_core_web_sm')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing functions
def remove_tags(text):
    patt = re.compile('<.*?>')
    return patt.sub('', text)

def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

def remove_chat_words(text):
    abbreviations = {}
    with open('slang.txt', 'r') as file:
        for line in file:
            line = line.strip()
            if '=' in line:
                abbreviation, expansion = line.split('=', 1)
                abbreviations[abbreviation] = expansion

    new_text = []
    for word in text.split():
        if word.upper() in abbreviations:
            new_text.append(abbreviations[word.upper()])
        else:
            new_text.append(word)
    return " ".join(new_text)

def apply_textblob_correction(value):
    blob = TextBlob(value)
    corrected_value = str(blob.correct())
    return corrected_value

def remove_stop_words(text):
    new_text = []
    for word in text.split():
        if word not in stop_words:
            new_text.append(word)
    return " ".join(new_text)

def remove_emojis(text):
    return ''.join(ch for ch in text if ch not in emoji.UNICODE_EMOJI['en'])

def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = [token.lemma_ for token in doc]
    return " ".join(lemmatized_text)

def preprocess_text(text):
    text = text.lower()
    text = remove_tags(text)
    text = remove_punc(text)
    text = remove_chat_words(text)
    text = apply_textblob_correction(text)
    text = remove_stop_words(text)
    text = lemmatize_text(text)
    return text

def predict_class(prediction):
    class_mapping = {
        0: 'not_flagged',
        1: 'flagged'
    }
    return class_mapping[prediction]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the HTML form
    text = request.form['text']

    # Preprocess the input text
    processed_text = preprocess_text(text)

    # Apply the CountVectorizer to the preprocessed text
    text_vector = vectorizer.transform([processed_text])

    # Make the prediction using the trained model
    prediction = model.predict(text_vector)
    class_prediction = predict_class(prediction[0])

    # Render the prediction result on the result.html template
    return render_template('result.html', prediction=class_prediction)

if __name__ == '__main__':
    app.run(debug=True)
