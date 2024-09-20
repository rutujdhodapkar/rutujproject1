from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatization and stopword removal
    return text

# Load the dataset and cache it
@st.cache_data
def load_data(data_path):
    return pd.read_csv(data_path)

# Load the dataset
data_path = 'Emotion_final_with_predictions.csv'  # Ensure this path is correct for deployment
data = load_data(data_path)

# Prepare data
X = data['Text'].apply(preprocess_text)  # Apply preprocessing
y = data['Emotion']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for vectorization and model training
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('logreg', LogisticRegression(max_iter=1000))
])

# Hyperparameter tuning
param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10, 100],
    'logreg__solver': ['lbfgs', 'liblinear']
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Prediction function
def predict_emotion(text):
    return best_model.predict([preprocess_text(text)])[0]

# Streamlit app
st.title('Text to Emotion Prediction')

# Display the dataset with only one row per emotion
st.subheader('Dataset (Example)')
unique_emotions = data.groupby('Emotion').first().reset_index()
st.write(unique_emotions)

# Input for prediction
st.subheader('Predict Emotion from Text')
st.markdown("Note: This is an advanced ML model. Use clear, relevant text for better predictions.")
input_text = st.text_input('Enter text:')

if st.button('Predict'):
    if input_text:
        predicted_emotion = predict_emotion(input_text)
        st.write(f'Predicted Emotion: **{predicted_emotion}**')
    else:
        st.write('Please enter some text to predict the emotion.')

# Add credits and note
st.markdown("**App developed by: Rutuj Dhodapkar**")
st.markdown("Note-1: Ensure your sentences are not based on your current situation; the model does not use real-time data.")
st.markdown("Note-2: Verify your sentence with other sources to understand the correct output.")
st.markdown("Note-3: This application uses an advanced machine learning model to predict emotions based on input text. For optimal performance, provide relevant and clear text (no special characters) to ensure the most accurate predictions.")
