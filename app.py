from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import json
import nltk
import spacy
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

app = Flask(__name__)

# Load NLP tools and models
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")

# Load the dataset and train the model
try:
    df = pd.read_csv("C:/Users/Learner/Downloads/mydataset.csv", sep=";", names=["Description", "Emotion"])
    df['preprocessed_text'] = df['Description'].apply(
        lambda x: " ".join([token.lemma_ for token in nlp(x) if not token.is_stop and not token.is_punct])
    )

    # Encode emotions as numerical labels
    label_encoder = LabelEncoder()
    df['Emotion_label'] = label_encoder.fit_transform(df['Emotion'])

    # Train the model
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', RandomForestClassifier())
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        df['preprocessed_text'], 
        df['Emotion_label'], 
        test_size=0.25, 
        random_state=42, 
        stratify=df['Emotion_label']
    )
    pipeline.fit(X_train, y_train)

    # Predict the labels for the test set
    y_pred = pipeline.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 7))
    cm_display.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')  # Save the confusion matrix image
    plt.close()  # Close the plot to prevent displaying it in Colab

except Exception as e:
    print(f"Error loading dataset or training model: {e}")

# Fetch response based on classified emotion
def get_response(emotion):
    try:
        with open(r'D:/code tools/CCP-3rd-sem/responses.json', 'r') as file:
            responses = json.load(file)
        return responses.get(emotion, "I am not sure how to respond to that.")
    except FileNotFoundError:
        return "Response file not found."
    except json.JSONDecodeError:
        return "Error decoding the response file."
    except Exception as e:
        return f"Error: {str(e)}"

# Classify emotion
def classify_emotion(text):
    sentiment = sia.polarity_scores(text)
    compound_score = sentiment['compound']
    
    if compound_score >= 0.05:
        return 'happy'
    elif compound_score >= 0.03:
        return 'surprised'
    elif compound_score >= 0.01:
        return 'hopeful'
    elif compound_score >= 0:
        return 'relieved'
    elif compound_score >= -0.05:
        return 'sad'
    elif compound_score < -0.05:
        return 'angry'
    else:
        return 'neutral'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get_bot_response():
    user_message = request.form['msg']
    emotion = classify_emotion(user_message)
    bot_response = get_response(emotion)
    return jsonify({'response': bot_response})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
